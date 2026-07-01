#!/usr/bin/env python3
"""Phase4m temporal mask-view local2history repair.

This repair keeps Phase2c local-window rows untouched and uses only GT-free
overlap/nearby-frame mask-view evidence to stitch chunk-local F2 objects into
scene identities. Exact same-frame mask matches were already tested by Phase4h;
the new evidence here is binary mask IoU between nearby frames across adjacent
chunks, optionally gated by semantic residual similarity.
"""

from __future__ import annotations

import csv
import json
import math
import sys
import time
from collections import OrderedDict, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


STREAM3D_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = STREAM3D_ROOT.parent
if str(STREAM3D_ROOT) not in sys.path:
    sys.path.insert(0, str(STREAM3D_ROOT))

from tools import build_v100_phase4h_overlap3_exact_history_memory as p4h  # noqa: E402
from tools import build_v99_phase1_f2_base_reproduction as p1  # noqa: E402


AUDIT_ROOT = STREAM3D_ROOT / "outputs/audit"
OUT_DIR = AUDIT_ROOT / "v100_phase4m_temporal_maskview_history_memory"
PHASE2C_DIR = AUDIT_ROOT / "v100_phase2c_overlap3_local_repair"
PHASE2C_SUMMARY = PHASE2C_DIR / "summary.json"
PHASE4H_DIR = AUDIT_ROOT / "v100_phase4h_overlap3_exact_history_memory"
PHASE4H_INTERNAL = PHASE4H_DIR / "internal_overlap_mv_object_frame_mask_rows.parquet"
PHASE4H_OVERLAP_ROWS = PHASE4H_DIR / "overlap_pair_rows.csv"

CHUNK_SIZE = p4h.CHUNK_SIZE
OVERLAP = p4h.OVERLAP


_LABEL_CACHE: OrderedDict[tuple[str, tuple[int, int] | None], np.ndarray] = OrderedDict()
_LABEL_CACHE_LIMIT = 12
_IOU_CACHE: dict[tuple[str, int, int, int, int], dict[str, Any]] = {}


def _rel(path: Path | str) -> str:
    return p4h._rel(path)


def _num(value: Any, default: float = 0.0) -> float:
    return p4h._num(value, default)


def _bool(value: Any) -> bool:
    return p4h._bool(value)


def _jsonable(value: Any) -> Any:
    return p4h._jsonable(value)


def _write_json(path: Path, payload: Any) -> None:
    p4h._write_json(path, payload)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    p4h._write_csv(path, rows)


def _write_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    p4h._write_parquet(path, rows)


def _cached_label(path: Path, shape_hw: tuple[int, int] | None = None) -> np.ndarray:
    key = (path.as_posix(), shape_hw)
    arr = _LABEL_CACHE.get(key)
    if arr is not None:
        _LABEL_CACHE.move_to_end(key)
        return arr
    arr = p1._read_label(path, shape_hw)
    _LABEL_CACHE[key] = arr
    _LABEL_CACHE.move_to_end(key)
    while len(_LABEL_CACHE) > _LABEL_CACHE_LIMIT:
        _LABEL_CACHE.popitem(last=False)
    return arr


def _mask_pair_iou(
    scope: dict[str, Any],
    *,
    scene: str,
    frame_a: int,
    mask_a: int,
    frame_b: int,
    mask_b: int,
) -> dict[str, Any]:
    cache_key = (scene, int(frame_a), int(mask_a), int(frame_b), int(mask_b))
    if cache_key in _IOU_CACHE:
        return _IOU_CACHE[cache_key]
    reverse_key = (scene, int(frame_b), int(mask_b), int(frame_a), int(mask_a))
    if reverse_key in _IOU_CACHE:
        return _IOU_CACHE[reverse_key]

    path_a = scope["mask_path_by_frame"].get((scene, int(frame_a)))
    path_b = scope["mask_path_by_frame"].get((scene, int(frame_b)))
    if path_a is None or path_b is None or not Path(path_a).exists() or not Path(path_b).exists():
        out = {
            "valid": False,
            "iou": 0.0,
            "intersection": 0,
            "union": 0,
            "area_a": 0,
            "area_b": 0,
            "missing_mask_path": True,
        }
        _IOU_CACHE[cache_key] = out
        return out

    label_a = _cached_label(Path(path_a))
    shape_hw = tuple(int(v) for v in label_a.shape[:2])
    label_b = _cached_label(Path(path_b), shape_hw)

    if int(frame_a) == int(frame_b):
        if int(mask_a) != int(mask_b):
            ma = label_a == int(mask_a)
            mb = label_b == int(mask_b)
            area_a = int(np.count_nonzero(ma))
            area_b = int(np.count_nonzero(mb))
            out = {
                "valid": area_a > 0 and area_b > 0,
                "iou": 0.0,
                "intersection": 0,
                "union": int(area_a + area_b),
                "area_a": area_a,
                "area_b": area_b,
                "missing_mask_path": False,
            }
            _IOU_CACHE[cache_key] = out
            return out
        ma = label_a == int(mask_a)
        area = int(np.count_nonzero(ma))
        out = {
            "valid": area > 0,
            "iou": 1.0 if area > 0 else 0.0,
            "intersection": area,
            "union": area,
            "area_a": area,
            "area_b": area,
            "missing_mask_path": False,
        }
        _IOU_CACHE[cache_key] = out
        return out

    ma = label_a == int(mask_a)
    mb = label_b == int(mask_b)
    area_a = int(np.count_nonzero(ma))
    area_b = int(np.count_nonzero(mb))
    if area_a == 0 or area_b == 0:
        out = {
            "valid": False,
            "iou": 0.0,
            "intersection": 0,
            "union": int(area_a + area_b),
            "area_a": area_a,
            "area_b": area_b,
            "missing_mask_path": False,
        }
        _IOU_CACHE[cache_key] = out
        return out
    inter = int(np.count_nonzero(ma & mb))
    union = int(area_a + area_b - inter)
    out = {
        "valid": union > 0,
        "iou": float(inter / union) if union > 0 else 0.0,
        "intersection": inter,
        "union": union,
        "area_a": area_a,
        "area_b": area_b,
        "missing_mask_path": False,
    }
    _IOU_CACHE[cache_key] = out
    return out


def _load_internal_rows() -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]], str]:
    if PHASE4H_INTERNAL.exists() and PHASE4H_OVERLAP_ROWS.exists():
        df = pd.read_parquet(PHASE4H_INTERNAL)
        internal = {
            split: [dict(row) for row in sub.to_dict(orient="records")]
            for split, sub in df.groupby("dataset_split")
        }
        with PHASE4H_OVERLAP_ROWS.open(newline="", encoding="utf-8") as f:
            overlap_rows = [dict(row) for row in csv.DictReader(f)]
        return internal, overlap_rows, _rel(PHASE4H_INTERNAL)

    payloads = {split: p4h._regenerate_internal_rows(split) for split in ["dev", "holdout"]}
    internal = {split: payload[0] for split, payload in payloads.items()}
    overlap_rows = [row for payload in payloads.values() for row in payload[2]]
    return internal, overlap_rows, "regenerated_in_phase4m"


def _top_temporal_mask_evidence(
    scope: dict[str, Any],
    info_a: dict[str, Any],
    info_b: dict[str, Any],
    *,
    nearby_frame_gap: int,
) -> dict[str, Any]:
    scene = str(info_a["scene_id"])
    exact_matches = set(info_a["frame_masks"]) & set(info_b["frame_masks"])
    temporal_values: list[dict[str, Any]] = []
    valid_pair_count = 0
    missing_pair_count = 0

    for frame_a, mask_a in sorted(info_a["frame_masks"]):
        for frame_b, mask_b in sorted(info_b["frame_masks"]):
            frame_gap = abs(int(frame_a) - int(frame_b))
            if frame_gap > nearby_frame_gap:
                continue
            stats = _mask_pair_iou(
                scope,
                scene=scene,
                frame_a=int(frame_a),
                mask_a=int(mask_a),
                frame_b=int(frame_b),
                mask_b=int(mask_b),
            )
            if stats.get("missing_mask_path"):
                missing_pair_count += 1
            if not stats.get("valid"):
                continue
            valid_pair_count += 1
            temporal_values.append(
                {
                    "frame_a": int(frame_a),
                    "mask_a": int(mask_a),
                    "frame_b": int(frame_b),
                    "mask_b": int(mask_b),
                    "frame_gap": int(frame_gap),
                    "iou": float(stats["iou"]),
                    "intersection": int(stats["intersection"]),
                    "union": int(stats["union"]),
                    "area_a": int(stats["area_a"]),
                    "area_b": int(stats["area_b"]),
                }
            )

    temporal_values.sort(key=lambda item: (item["iou"], -item["frame_gap"], item["intersection"]), reverse=True)
    top = temporal_values[:3]
    max_iou = float(top[0]["iou"]) if top else 0.0
    mean_top3 = float(np.mean([item["iou"] for item in top])) if top else 0.0
    best = top[0] if top else {}
    return {
        "exact_overlap_frame_mask_count": len(exact_matches),
        "exact_overlap_frame_count": len({frame for frame, _mask in exact_matches}),
        "nearby_temporal_pair_count": len(temporal_values),
        "valid_nearby_temporal_pair_count": valid_pair_count,
        "missing_temporal_pair_count": missing_pair_count,
        "max_temporal_mask_iou": max_iou,
        "mean_top3_temporal_mask_iou": mean_top3,
        "best_temporal_frame_a": best.get("frame_a", ""),
        "best_temporal_mask_a": best.get("mask_a", ""),
        "best_temporal_frame_b": best.get("frame_b", ""),
        "best_temporal_mask_b": best.get("mask_b", ""),
        "best_temporal_frame_gap": best.get("frame_gap", ""),
        "best_temporal_intersection": best.get("intersection", ""),
        "best_temporal_union": best.get("union", ""),
    }


def _candidate_rows(
    infos: dict[str, dict[str, Any]],
    features: dict[tuple[str, int, int], np.ndarray],
    scope: dict[str, Any],
    *,
    variant_id: str,
    min_temporal_iou: float,
    nearby_frame_gap: int,
    semantic_floor: float,
) -> list[dict[str, Any]]:
    del features
    by_scene_chunk: dict[tuple[str, str], list[str]] = defaultdict(list)
    chunks_by_scene: dict[str, set[str]] = defaultdict(set)
    for oid, info in infos.items():
        scene = str(info["scene_id"])
        chunk = str(info["chunk_id"])
        by_scene_chunk[(scene, chunk)].append(oid)
        chunks_by_scene[scene].add(chunk)

    rows: list[dict[str, Any]] = []
    for scene, chunk_set in sorted(chunks_by_scene.items()):
        chunks = sorted(chunk_set, key=p4h._chunk_index)
        for left, right in zip(chunks[:-1], chunks[1:]):
            for a in sorted(by_scene_chunk[(scene, left)]):
                fa = infos[a].get("feature")
                for b in sorted(by_scene_chunk[(scene, right)]):
                    fb = infos[b].get("feature")
                    sem = float(np.dot(fa, fb)) if fa is not None and fb is not None else 0.0
                    evidence = _top_temporal_mask_evidence(
                        scope,
                        infos[a],
                        infos[b],
                        nearby_frame_gap=nearby_frame_gap,
                    )
                    exact_count = int(evidence["exact_overlap_frame_mask_count"])
                    max_iou = float(evidence["max_temporal_mask_iou"])
                    temporal_ok = max_iou >= min_temporal_iou
                    semantic_ok = sem >= semantic_floor
                    if exact_count > 0:
                        family = "exact_overlap_frame_mask"
                        affinity = float(1.0 + 0.01 * exact_count + 0.001 * sem)
                    elif temporal_ok and semantic_ok:
                        family = f"temporal_maskview_iou{min_temporal_iou:.2f}_sem{semantic_floor:.2f}"
                        affinity = float(max_iou + 0.25 * evidence["mean_top3_temporal_mask_iou"] + 0.01 * sem)
                    else:
                        continue
                    rows.append(
                        {
                            "schema_version": "stream4d_v100_phase4m_local2history_candidate_row_v1",
                            "phase_id": "v100_phase4m_temporal_maskview_history_memory",
                            "variant_id": variant_id,
                            "dataset_split": infos[a]["dataset_split"],
                            "scene_id": scene,
                            "left_chunk_id": left,
                            "right_chunk_id": right,
                            "chunk_gap": 1,
                            "mv_object_id_a": a,
                            "mv_object_id_b": b,
                            "candidate_family": family,
                            "semantic_cosine": sem,
                            "min_temporal_iou": min_temporal_iou,
                            "semantic_floor": semantic_floor,
                            "nearby_frame_gap": nearby_frame_gap,
                            "affinity": affinity,
                            "uses_gt_for_prediction": False,
                            "uses_future": False,
                            **evidence,
                        }
                    )
    return rows


def _one_to_one_mapping(ids: list[str], candidates: list[dict[str, Any]], *, variant_id: str) -> tuple[dict[str, str], list[dict[str, Any]]]:
    dsu = p4h.DSU(ids)
    used_left: set[tuple[str, str, str]] = set()
    used_right: set[tuple[str, str, str]] = set()
    accepted: list[dict[str, Any]] = []
    for row in sorted(
        candidates,
        key=lambda item: (
            _num(item.get("affinity")),
            _num(item.get("exact_overlap_frame_mask_count")),
            _num(item.get("max_temporal_mask_iou")),
            _num(item.get("mean_top3_temporal_mask_iou")),
            _num(item.get("semantic_cosine")),
            -_num(item.get("chunk_gap")),
            str(item.get("mv_object_id_a")),
            str(item.get("mv_object_id_b")),
        ),
        reverse=True,
    ):
        a = str(row.get("mv_object_id_a"))
        b = str(row.get("mv_object_id_b"))
        scene = str(row.get("scene_id"))
        left_chunk = str(row.get("left_chunk_id"))
        right_chunk = str(row.get("right_chunk_id"))
        left_key = (scene, left_chunk, a)
        right_key = (scene, right_chunk, b)
        if a not in dsu.parent or b not in dsu.parent or left_key in used_left or right_key in used_right:
            continue
        if dsu.union(a, b):
            new = dict(row)
            new["schema_version"] = "stream4d_v100_phase4m_local2history_merge_row_v1"
            new["phase_id"] = "v100_phase4m_temporal_maskview_history_memory"
            new["variant_id"] = variant_id
            new["merge_policy"] = "causal_adjacent_overlap3_one_to_one_temporal_maskview_greedy"
            accepted.append(new)
            used_left.add(left_key)
            used_right.add(right_key)
    mapping = {oid: f"{variant_id}:{dsu.find(oid)}" for oid in ids}
    return mapping, accepted


def _apply_mapping(rows: list[dict[str, Any]], mapping: dict[str, str], *, variant_id: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        oid = str(row["mv_object_id"])
        new = dict(row)
        new["schema_version"] = "stream4d_v100_phase4m_scene_mv_object_frame_mask_row_v1"
        new["phase_id"] = "v100_phase4m_temporal_maskview_history_memory"
        new["variant_id"] = variant_id
        new["variant"] = variant_id
        new["source_phase2c_mv_object_id"] = oid
        new["mv_object_id"] = mapping.get(oid, f"{variant_id}:{oid}")
        new["object_id"] = new["mv_object_id"]
        new["history_id"] = new["mv_object_id"]
        new["object_id_policy"] = "causal_temporal_maskview_history_stitched_identity"
        new["history_memory_scope"] = "causal_past_adjacent_chunks_overlap3_with_nearby_maskview_evidence"
        new["score_scope"] = "current_chunk_score_history_identity"
        new["uses_gt_for_prediction"] = False
        new["uses_future"] = False
        new["future_chunk_access"] = False
        out.append(new)
    return out


def _phase4m_artifact_rows(paths: list[tuple[Path, str, str]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path, kind, note in paths:
        rows.append(
            {
                "schema_version": "stream4d_v100_phase4m_artifact_manifest_row_v1",
                "phase_id": "v100_phase4m_temporal_maskview_history_memory",
                "artifact_path": _rel(path),
                "artifact_type": kind,
                "size_bytes": path.stat().st_size if path.exists() else 0,
                "sha256": p4h._sha256(path) if path.exists() and path.is_file() else "",
                "note": note,
            }
        )
    return rows


def main() -> int:
    started = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    phase2c = json.loads(PHASE2C_SUMMARY.read_text(encoding="utf-8"))
    if not bool(phase2c.get("phase2c_pass")):
        raise RuntimeError("Phase4m requires v100 Phase2c overlap3 local pass")

    baselines = p4h._phase0_baselines()
    f2_dev = baselines["F2_base_full_dev"]
    f2_holdout = baselines["F2_base_holdout"]

    primary_df = pd.read_parquet(PHASE2C_DIR / "mv_object_frame_mask_rows.parquet")
    primary_by_split = {
        split: [dict(row) for row in sub.to_dict(orient="records")]
        for split, sub in primary_df.groupby("dataset_split")
    }
    internal_by_split, overlap_rows, internal_source = _load_internal_rows()
    scopes = {split: p4h._scope_for_split(split) for split in ["dev", "holdout"]}
    features = {split: p4h._features_for_split(split) for split in ["dev", "holdout"]}

    primary_ids = {split: {str(row["mv_object_id"]) for row in rows} for split, rows in primary_by_split.items()}
    internal_ids = {split: {str(row["mv_object_id"]) for row in rows} for split, rows in internal_by_split.items()}
    missing_internal_id_count = {
        split: len(primary_ids[split] - internal_ids[split])
        for split in ["dev", "holdout"]
    }

    variant_specs = [
        {
            "variant_id": "HMV0_phase2c_fragmented_baseline",
            "family": "baseline",
            "min_temporal_iou": 2.0,
            "nearby_frame_gap": 0,
            "semantic_floor": 2.0,
            "candidate_mode": "none",
            "notes": "Phase2c primary rows with chunk-scoped ids; no history links.",
        },
        {
            "variant_id": "HMV1_exact_or_temporal_iou0p30_gap5",
            "family": "exact_or_temporal_maskview",
            "min_temporal_iou": 0.30,
            "nearby_frame_gap": 5,
            "semantic_floor": -1.0,
            "candidate_mode": "exact_or_temporal",
            "notes": "Adjacent chunks; exact overlap masks or strong nearby-frame mask IoU.",
        },
        {
            "variant_id": "HMV2_exact_or_temporal_iou0p20_gap10",
            "family": "exact_or_temporal_maskview",
            "min_temporal_iou": 0.20,
            "nearby_frame_gap": 10,
            "semantic_floor": -1.0,
            "candidate_mode": "exact_or_temporal",
            "notes": "Adjacent chunks; moderate nearby-frame mask IoU.",
        },
        {
            "variant_id": "HMV3_exact_or_temporal_iou0p10_gap15",
            "family": "exact_or_temporal_maskview",
            "min_temporal_iou": 0.10,
            "nearby_frame_gap": 15,
            "semantic_floor": -1.0,
            "candidate_mode": "exact_or_temporal",
            "notes": "Adjacent chunks; loose nearby-frame mask IoU.",
        },
        {
            "variant_id": "HMV4_exact_or_temporal_iou0p05_sem0p72_gap15",
            "family": "temporal_maskview_semantic_gated",
            "min_temporal_iou": 0.05,
            "nearby_frame_gap": 15,
            "semantic_floor": 0.72,
            "candidate_mode": "exact_or_temporal_and_semantic",
            "notes": "Adjacent chunks; weak nearby-frame mask IoU accepted only with semantic residual agreement.",
        },
    ]

    variant_metric_rows: list[dict[str, Any]] = []
    scene_metric_rows: list[dict[str, Any]] = []
    frame_eval_rows: list[dict[str, Any]] = []
    candidate_rows_all: list[dict[str, Any]] = []
    merge_rows_all: list[dict[str, Any]] = []
    config_rows: list[dict[str, Any]] = []
    rows_by_variant_split: dict[tuple[str, str], list[dict[str, Any]]] = {}

    infos_by_split = {
        split: p4h._object_infos(internal_by_split[split], features[split])
        for split in ["dev", "holdout"]
    }

    for spec in variant_specs:
        variant_id = str(spec["variant_id"])
        config_rows.append(
            {
                "schema_version": "stream4d_v100_phase4m_variant_config_row_v1",
                "phase_id": "v100_phase4m_temporal_maskview_history_memory",
                "variant_id": variant_id,
                "family": spec["family"],
                "min_temporal_iou": spec["min_temporal_iou"],
                "nearby_frame_gap": spec["nearby_frame_gap"],
                "semantic_floor": spec["semantic_floor"],
                "candidate_mode": spec["candidate_mode"],
                "notes": spec["notes"],
                "uses_gt_for_prediction": False,
                "uses_future": False,
            }
        )
        for split in ["dev", "holdout"]:
            primary_rows = primary_by_split[split]
            ids = sorted(primary_ids[split])
            if spec["candidate_mode"] == "none":
                mapping = {oid: f"{variant_id}:{oid}" for oid in ids}
                candidates: list[dict[str, Any]] = []
                accepted: list[dict[str, Any]] = []
            else:
                candidates = _candidate_rows(
                    infos_by_split[split],
                    features[split],
                    scopes[split],
                    variant_id=variant_id,
                    min_temporal_iou=float(spec["min_temporal_iou"]),
                    nearby_frame_gap=int(spec["nearby_frame_gap"]),
                    semantic_floor=float(spec["semantic_floor"]),
                )
                mapping, accepted = _one_to_one_mapping(ids, candidates, variant_id=variant_id)
            mapped_rows = _apply_mapping(primary_rows, mapping, variant_id=variant_id)
            rows_by_variant_split[(variant_id, split)] = mapped_rows
            candidate_rows_all.extend(candidates)
            merge_rows_all.extend(accepted)

            p4h._set_inputs(split)
            per_scene, frames = p1._evaluate_variant(variant_id, mapped_rows, scopes[split])
            scene_agg = p1._aggregate_metrics(per_scene)[0]
            local_agg = p4h._local_agg_from_phase2c(phase2c, split)
            component_stats = p4h._component_stats(mapping)
            crossing = p4h._scene_crossing_stats(mapped_rows)
            row = dict(scene_agg)
            row["schema_version"] = "stream4d_v100_phase4m_metric_aggregate_row_v1"
            row["phase_id"] = "v100_phase4m_temporal_maskview_history_memory"
            row["variant_id"] = variant_id
            row["dataset_split"] = split
            row["MV_AP_window_scene_id_scope"] = row.get("MV_AP_window")
            row["MV_AP50_window_scene_id_scope"] = row.get("MV_AP50_window")
            row["MV_AP_window"] = local_agg["MV_AP_window"]
            row["MV_AP50_window"] = local_agg["MV_AP50_window"]
            row["metric_composition"] = "local_window_from_phase2c_chunk_ids_scene_from_phase4m_stitched_ids"
            row["history_candidate_count"] = len(candidates)
            row["accepted_history_merge_edge_count"] = len(accepted)
            row["exact_overlap_candidate_count"] = sum(1 for c in candidates if c["candidate_family"] == "exact_overlap_frame_mask")
            row["temporal_maskview_candidate_count"] = sum(1 for c in candidates if c["candidate_family"] != "exact_overlap_frame_mask")
            row["max_temporal_mask_iou_seen"] = max([_num(c.get("max_temporal_mask_iou")) for c in candidates], default=0.0)
            row["mean_candidate_temporal_mask_iou"] = float(np.mean([_num(c.get("max_temporal_mask_iou")) for c in candidates])) if candidates else 0.0
            row["future_chunk_access"] = False
            row["uses_gt_for_prediction"] = False
            row["uses_future"] = False
            row.update(component_stats)
            row.update(crossing)
            variant_metric_rows.append(row)

            for item in per_scene:
                item["phase_id"] = "v100_phase4m_temporal_maskview_history_memory"
                item["dataset_split"] = split
                item["metric_scope_note"] = "scene id scope; aggregate row preserves Phase2c local-window ids"
            scene_metric_rows.extend(per_scene)
            for item in frames:
                item["phase_id"] = "v100_phase4m_temporal_maskview_history_memory"
                item["dataset_split"] = split
            frame_eval_rows.extend(frames)

    by_variant: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in variant_metric_rows:
        by_variant[str(row["variant_id"])][str(row["dataset_split"])] = row
    best_variant_id = max(
        by_variant,
        key=lambda vid: (
            _num(by_variant[vid].get("holdout", {}).get("MV_AP_scene")),
            _num(by_variant[vid].get("holdout", {}).get("MV_AP50_scene")),
            _num(by_variant[vid].get("dev", {}).get("MV_AP_scene")),
            _num(by_variant[vid].get("dev", {}).get("MV_AP50_scene")),
        ),
    )
    best_dev = by_variant[best_variant_id]["dev"]
    best_hold = by_variant[best_variant_id]["holdout"]
    best_rows = rows_by_variant_split[(best_variant_id, "dev")] + rows_by_variant_split[(best_variant_id, "holdout")]

    dev_scene_gate = _num(best_dev.get("MV_AP_scene")) >= _num(f2_dev["MV_AP_scene"]) + 0.010
    dev_scene_ap50_gate = _num(best_dev.get("MV_AP50_scene")) >= _num(f2_dev["MV_AP50_scene"]) + 0.015
    hold_scene_gate = _num(best_hold.get("MV_AP_scene")) >= _num(f2_holdout["MV_AP_scene"]) + 0.006
    hold_scene_ap50_gate = _num(best_hold.get("MV_AP50_scene")) >= _num(f2_holdout["MV_AP50_scene"]) + 0.010
    local_drop_dev = float(phase2c["dev_MV_AP_window"]) - _num(best_dev.get("MV_AP_window"))
    local_drop_hold = float(phase2c["holdout_MV_AP_window"]) - _num(best_hold.get("MV_AP_window"))
    local_drop_gate = local_drop_dev <= 0.003 and local_drop_hold <= 0.003
    objects_crossing_gate = int(_num(best_dev.get("objects_crossing_multiple_chunks"))) + int(_num(best_hold.get("objects_crossing_multiple_chunks"))) > 0
    safety_gate = (
        int(_num(best_dev.get("same_frame_collision_count"))) == 0
        and int(_num(best_hold.get("same_frame_collision_count"))) == 0
        and _num(best_dev.get("pixel_collision_rate")) <= 0.02
        and _num(best_hold.get("pixel_collision_rate")) <= 0.02
        and int(_num(best_dev.get("missing_mask_raster_count"))) == 0
        and int(_num(best_hold.get("missing_mask_raster_count"))) == 0
        and missing_internal_id_count["dev"] == 0
        and missing_internal_id_count["holdout"] == 0
    )
    overlap_gate = bool(overlap_rows) and all(int(float(row["shared_frame_count"])) == OVERLAP for row in overlap_rows)
    future_gate = not any(_bool(row.get("uses_future")) or _bool(row.get("future_chunk_access")) for row in best_rows)
    phase4m_pass = bool(
        dev_scene_gate
        and dev_scene_ap50_gate
        and hold_scene_gate
        and hold_scene_ap50_gate
        and local_drop_gate
        and objects_crossing_gate
        and safety_gate
        and overlap_gate
        and future_gate
    )

    gate_rows = [
        {
            "gate_id": "mv_ap_scene_dev_ge_f2_base_plus_0p010",
            "pass": dev_scene_gate,
            "expected": _num(f2_dev["MV_AP_scene"]) + 0.010,
            "observed": _num(best_dev.get("MV_AP_scene")),
            "severity": "required_scene",
        },
        {
            "gate_id": "mv_ap50_scene_dev_ge_f2_base_plus_0p015",
            "pass": dev_scene_ap50_gate,
            "expected": _num(f2_dev["MV_AP50_scene"]) + 0.015,
            "observed": _num(best_dev.get("MV_AP50_scene")),
            "severity": "required_scene",
        },
        {
            "gate_id": "mv_ap_scene_holdout_ge_f2_base_plus_0p006",
            "pass": hold_scene_gate,
            "expected": _num(f2_holdout["MV_AP_scene"]) + 0.006,
            "observed": _num(best_hold.get("MV_AP_scene")),
            "severity": "required_scene",
        },
        {
            "gate_id": "mv_ap50_scene_holdout_ge_f2_base_plus_0p010",
            "pass": hold_scene_ap50_gate,
            "expected": _num(f2_holdout["MV_AP50_scene"]) + 0.010,
            "observed": _num(best_hold.get("MV_AP50_scene")),
            "severity": "required_scene",
        },
        {
            "gate_id": "local_window_ap_drop_le_0p003",
            "pass": local_drop_gate,
            "expected": "<=0.003 for dev and holdout after local/scene scope separation",
            "observed": f"dev_drop={local_drop_dev}; holdout_drop={local_drop_hold}",
            "severity": "protect_local",
        },
        {
            "gate_id": "objects_crossing_multiple_chunks_gt_0",
            "pass": objects_crossing_gate,
            "expected": ">0",
            "observed": f"dev={best_dev.get('objects_crossing_multiple_chunks')} holdout={best_hold.get('objects_crossing_multiple_chunks')}",
            "severity": "identity_required",
        },
        {
            "gate_id": "overlap3_contract_shared_frames",
            "pass": overlap_gate,
            "expected": "all adjacent chunks shared_frame_count=3",
            "observed": sorted({int(float(row["shared_frame_count"])) for row in overlap_rows}),
            "severity": "formal_contract_required",
        },
        {
            "gate_id": "collision_missing_internal_id_safety",
            "pass": safety_gate,
            "expected": "collision=0 pixel<=0.02 missing_mask=0 and primary ids have internal witnesses",
            "observed": f"dev_collision={best_dev.get('same_frame_collision_count')} hold_collision={best_hold.get('same_frame_collision_count')} dev_missing={best_dev.get('missing_mask_raster_count')} hold_missing={best_hold.get('missing_mask_raster_count')} missing_internal_ids={missing_internal_id_count}",
            "severity": "required_safety",
        },
        {
            "gate_id": "future_chunk_access_false",
            "pass": future_gate,
            "expected": "false for all best rows",
            "observed": future_gate,
            "severity": "required_safety",
        },
    ]
    failure_rows = [
        {
            "schema_version": "stream4d_v100_phase4m_failure_row_v1",
            "phase_id": "v100_phase4m_temporal_maskview_history_memory",
            "failure_id": row["gate_id"],
            "severity": row["severity"],
            "expected": row["expected"],
            "observed": row["observed"],
            "repair_direction": (
                "If temporal mask-view candidates undermerge, inspect whether support is absent around overlap and test a geometry/provider verifier. "
                "If candidates overmerge, add cannot-link from overlap conflict or restrict to high-confidence temporal support."
            ),
        }
        for row in gate_rows
        if not bool(row["pass"])
    ]

    scene_frame_parquet = OUT_DIR / "scene_mv_object_frame_mask_rows.parquet"
    candidate_csv = OUT_DIR / "local2history_candidate_rows.csv"
    merge_csv = OUT_DIR / "local2history_merge_rows.csv"
    metric_csv = OUT_DIR / "variant_metric_rows.csv"
    scene_metric_csv = OUT_DIR / "mv_metric_scene_rows.csv"
    frame_csv = OUT_DIR / "frame_eval_rows.csv"
    config_csv = OUT_DIR / "variant_config_rows.csv"
    gate_csv = OUT_DIR / "variant_gate_rows.csv"
    failure_csv = OUT_DIR / "variant_failure_rows.csv"
    performance_csv = OUT_DIR / "performance_rows.csv"
    artifact_csv = OUT_DIR / "artifact_manifest_rows.csv"

    _write_parquet(scene_frame_parquet, best_rows)
    _write_csv(candidate_csv, candidate_rows_all)
    _write_csv(merge_csv, merge_rows_all)
    _write_csv(metric_csv, variant_metric_rows)
    _write_csv(scene_metric_csv, scene_metric_rows)
    _write_csv(frame_csv, frame_eval_rows)
    _write_csv(config_csv, config_rows)
    _write_csv(gate_csv, gate_rows)
    _write_csv(failure_csv, failure_rows)
    _write_csv(
        performance_csv,
        [
            {
                "schema_version": "stream4d_v100_phase4m_performance_row_v1",
                "phase_id": "v100_phase4m_temporal_maskview_history_memory",
                "case_id": "temporal_maskview_candidate_generation_and_v65_eval",
                "runtime_sec": time.time() - started,
                "variant_count": len(variant_specs),
                "split_count": 2,
                "v65_evaluator_runs": len(variant_specs) * 2,
                "internal_row_count": sum(len(rows) for rows in internal_by_split.values()),
                "candidate_count": len(candidate_rows_all),
                "merge_count": len(merge_rows_all),
                "mask_iou_cache_entries": len(_IOU_CACHE),
                "label_cache_limit": _LABEL_CACHE_LIMIT,
                "internal_source": internal_source,
            }
        ],
    )
    _write_csv(
        artifact_csv,
        _phase4m_artifact_rows(
            [
                (scene_frame_parquet, "parquet", "best variant primary-emitted scene rows"),
                (candidate_csv, "csv", "temporal mask-view local2history candidate rows"),
                (merge_csv, "csv", "accepted local2history merge rows"),
                (metric_csv, "csv", "aggregate metrics for all variants/splits"),
                (scene_metric_csv, "csv", "v65 per-scene metrics"),
                (frame_csv, "csv", "v65 frame eval rows"),
                (config_csv, "csv", "variant configs"),
                (gate_csv, "csv", "phase4m gates"),
                (failure_csv, "csv", "phase4m failures if any"),
                (performance_csv, "csv", "runtime and row counts"),
            ]
        ),
    )

    summary = {
        "schema_version": "stream4d_v100_phase4m_temporal_maskview_history_memory_summary_v1",
        "phase_id": "v100_phase4m_temporal_maskview_history_memory",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "runtime_sec": time.time() - started,
        "decision": "PASS_ENTER_PHASE5" if phase4m_pass else "BLOCK_PHASE5_REPAIR_TEMPORAL_MASKVIEW_HISTORY_MEMORY",
        "phase4m_pass": phase4m_pass,
        "failure_count": len(failure_rows),
        "best_variant_id": best_variant_id,
        "best_dev_MV_AP_window": float(_num(best_dev.get("MV_AP_window"))),
        "best_dev_MV_AP50_window": float(_num(best_dev.get("MV_AP50_window"))),
        "best_dev_MV_AP_scene": float(_num(best_dev.get("MV_AP_scene"))),
        "best_dev_MV_AP50_scene": float(_num(best_dev.get("MV_AP50_scene"))),
        "best_holdout_MV_AP_window": float(_num(best_hold.get("MV_AP_window"))),
        "best_holdout_MV_AP50_window": float(_num(best_hold.get("MV_AP50_window"))),
        "best_holdout_MV_AP_scene": float(_num(best_hold.get("MV_AP_scene"))),
        "best_holdout_MV_AP50_scene": float(_num(best_hold.get("MV_AP50_scene"))),
        "local_window_AP_drop": {"dev": local_drop_dev, "holdout": local_drop_hold},
        "objects_crossing_multiple_chunks": {
            "dev": int(_num(best_dev.get("objects_crossing_multiple_chunks"))),
            "holdout": int(_num(best_hold.get("objects_crossing_multiple_chunks"))),
        },
        "accepted_history_merge_edge_count": {
            "dev": int(_num(best_dev.get("accepted_history_merge_edge_count"))),
            "holdout": int(_num(best_hold.get("accepted_history_merge_edge_count"))),
        },
        "temporal_maskview_candidate_count": {
            "dev": int(_num(best_dev.get("temporal_maskview_candidate_count"))),
            "holdout": int(_num(best_hold.get("temporal_maskview_candidate_count"))),
        },
        "exact_overlap_candidate_count": {
            "dev": int(_num(best_dev.get("exact_overlap_candidate_count"))),
            "holdout": int(_num(best_hold.get("exact_overlap_candidate_count"))),
        },
        "max_temporal_mask_iou_seen": {
            "dev": float(_num(best_dev.get("max_temporal_mask_iou_seen"))),
            "holdout": float(_num(best_hold.get("max_temporal_mask_iou_seen"))),
        },
        "max_component_size": {
            "dev": int(_num(best_dev.get("max_component_size"))),
            "holdout": int(_num(best_hold.get("max_component_size"))),
        },
        "overlap_contract_gate_pass": overlap_gate,
        "missing_internal_id_count": missing_internal_id_count,
        "internal_source": internal_source,
        "mask_iou_cache_entries": len(_IOU_CACHE),
        "future_chunk_access": False,
        "uses_gt_for_prediction": False,
        "metric_composition": "local_window_from_phase2c_chunk_ids_scene_from_phase4m_stitched_ids",
        "outputs": {
            "summary": _rel(OUT_DIR / "summary.json"),
            "scene_mv_object_frame_mask_rows": _rel(scene_frame_parquet),
            "local2history_candidate_rows": _rel(candidate_csv),
            "local2history_merge_rows": _rel(merge_csv),
            "variant_metric_rows": _rel(metric_csv),
            "mv_metric_scene_rows": _rel(scene_metric_csv),
            "frame_eval_rows": _rel(frame_csv),
            "variant_config_rows": _rel(config_csv),
            "variant_gate_rows": _rel(gate_csv),
            "variant_failure_rows": _rel(failure_csv),
            "performance_rows": _rel(performance_csv),
            "artifact_manifest_rows": _rel(artifact_csv),
        },
    }
    _write_json(OUT_DIR / "summary.json", summary)
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))
    return 0 if phase4m_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
