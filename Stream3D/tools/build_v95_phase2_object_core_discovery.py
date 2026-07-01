#!/usr/bin/env python3
"""Discover v95 object cores from object-axis unary shards."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/audit/v95_phase2_object_core_discovery"
PHASE_ID = "v95_phase2_object_core_discovery"
RUN_ID = "v95_phase2_object_core_discovery"

PHASE1 = ROOT / "outputs/audit/v95_phase1_physical_source_registry"
DEFAULT_FIELD_ROOT = ROOT / "outputs/audit/v94_phase7c_object_axis_field_full_dev_combined"


def _created_at() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return value.as_posix()
    return value


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _jsonable(row.get(key, "")) for key in fields})


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def _num(value: Any, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return float(default)
        out = float(value)
    except Exception:
        return float(default)
    return out if math.isfinite(out) else float(default)


def _int(value: Any, default: int = -1) -> int:
    try:
        if value in (None, ""):
            return int(default)
        return int(float(value))
    except Exception:
        return int(default)


def _bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _mean(values: list[float]) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    return float(np.mean(np.asarray(vals, dtype=np.float64))) if vals else 0.0


def _percentile(values: list[float], q: float) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    return float(np.percentile(np.asarray(vals, dtype=np.float64), q)) if vals else 0.0


def _source_key_parts(raw: str) -> tuple[str, str, int, int]:
    scene, window, frame_raw, mask_raw = str(raw).split("|")
    return scene, window, int(frame_raw), int(mask_raw)


def _source_key_text(key: tuple[str, str, int, int]) -> str:
    scene, window, frame_id, mask_id = key
    return f"{scene}|{window}|{frame_id}|{mask_id}"


def _variant_specs() -> list[dict[str, Any]]:
    specs = [
        {
            "variant_id": "CORE_A_d4rt_witness_strict",
            "core_family": "D4RT_witness_core",
            "mode": "d4rt",
            "top_fraction": 0.04,
            "max_regions": 8,
            "min_score": 0.58,
            "max_risk": 0.35,
            "min_margin": 0.015,
            "max_area_ratio": 0.30,
        },
        {
            "variant_id": "CORE_B_radio_medoid_topk",
            "core_family": "RADIO_medoid_core",
            "mode": "radio",
            "top_fraction": 0.035,
            "max_regions": 7,
            "min_score": 0.55,
            "max_risk": 0.40,
            "min_margin": 0.025,
            "max_area_ratio": 0.28,
        },
        {
            "variant_id": "CORE_C_d4rt_radio_consensus",
            "core_family": "D4RT_RADIO_consensus_core",
            "mode": "consensus",
            "top_fraction": 0.03,
            "max_regions": 6,
            "min_score": 0.60,
            "max_risk": 0.32,
            "min_margin": 0.020,
            "max_area_ratio": 0.24,
        },
        {
            "variant_id": "CORE_D_low_risk_repeated_support",
            "core_family": "low_risk_repeated_support_core",
            "mode": "repeated",
            "top_fraction": 0.025,
            "max_regions": 5,
            "min_score": 0.52,
            "max_risk": 0.35,
            "min_margin": 0.015,
            "max_area_ratio": 0.22,
            "min_repeat_count": 2,
        },
        {
            "variant_id": "CORE_E_hard_negative_separated",
            "core_family": "hard_negative_separated_core",
            "mode": "separated",
            "top_fraction": 0.03,
            "max_regions": 6,
            "min_score": 0.50,
            "max_risk": 0.45,
            "min_margin": 0.045,
            "max_area_ratio": 0.25,
        },
    ]
    for spec in specs:
        spec.update(
            {
                "area_repair_mode": "radio_consistent_neighbor_expansion",
                "expansion_target_area_ratio": 0.022,
                "expansion_max_extra_regions": 4,
                "expansion_min_radio": 0.50,
                "expansion_min_margin": 0.006,
                "expansion_min_edge_radio": 0.82,
                "expansion_radio_drop": 0.12,
                "expansion_max_risk_delta": 0.08,
            }
        )
    return specs


def _load_phase1_sources() -> tuple[dict[tuple[str, str, int, int], dict[str, str]], dict[str, int]]:
    sources: dict[tuple[str, str, int, int], dict[str, str]] = {}
    with (PHASE1 / "source_container_rows.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = (row["scene_id"], row["window_id"], _int(row["frame_id"]), _int(row["source_mask_id"]))
            sources[key] = dict(row)
    repeats: Counter[str] = Counter()
    with (PHASE1 / "source_object_registry_rows.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if _bool(row.get("uses_gt_for_prediction")) or _bool(row.get("uses_future")):
                continue
            repeats[str(row.get("object_id", ""))] += 1
    return sources, dict(repeats)


def _collect_shard_source_keys(shard_paths: list[Path], max_shards: int, max_sources: int) -> list[str]:
    out: list[str] = []
    for shard_path in shard_paths[: max_shards if max_shards > 0 else None]:
        with np.load(shard_path, allow_pickle=False) as data:
            out.extend(str(v) for v in data["source_keys"].tolist())
        if max_sources > 0 and len(out) >= max_sources:
            return out[:max_sources]
    return out


def _load_region_nodes(path: Path, selected_keys: set[tuple[str, str, int, int]]) -> dict[tuple[str, str, int, int], dict[int, dict[str, str]]]:
    nodes: dict[tuple[str, str, int, int], dict[int, dict[str, str]]] = defaultdict(dict)
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = (row["scene_id"], row["window_id"], _int(row["frame_id"]), _int(row["source_mask_id"]))
            if key not in selected_keys:
                continue
            if _bool(row.get("uses_gt_for_prediction")) or _bool(row.get("uses_future")):
                continue
            nodes[key][_int(row.get("region_index"))] = dict(row)
    return dict(nodes)


def _load_region_edges(path: Path, selected_keys: set[tuple[str, str, int, int]]) -> dict[tuple[str, str, int, int], list[dict[str, str]]]:
    edges: dict[tuple[str, str, int, int], list[dict[str, str]]] = defaultdict(list)
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = (row["scene_id"], row["window_id"], _int(row["frame_id"]), _int(row["source_mask_id"]))
            if key not in selected_keys:
                continue
            if _bool(row.get("uses_gt_for_prediction")) or _bool(row.get("uses_future")):
                continue
            if not _bool(row.get("is_adjacent")):
                continue
            edges[key].append(dict(row))
    return dict(edges)


def _source_adjacency(edge_rows: list[dict[str, str]], region_id_to_local: dict[str, int]) -> dict[int, list[tuple[int, float, float]]]:
    adjacency: dict[int, list[tuple[int, float, float]]] = defaultdict(list)
    for row in edge_rows:
        u_raw = str(row.get("region_u") or row.get("region_id_a") or "")
        v_raw = str(row.get("region_v") or row.get("region_id_b") or "")
        if u_raw not in region_id_to_local or v_raw not in region_id_to_local:
            continue
        u = int(region_id_to_local[u_raw])
        v = int(region_id_to_local[v_raw])
        if u == v:
            continue
        edge_radio = max(0.0, _num(row.get("radio_cosine"), 0.0))
        edge_weight = max(0.0, _num(row.get("edge_weight"), edge_radio))
        adjacency[u].append((v, edge_radio, edge_weight))
        adjacency[v].append((u, edge_radio, edge_weight))
    return dict(adjacency)


def _node_features(nodes: list[dict[str, str]], source_area: float) -> dict[str, np.ndarray]:
    area = np.asarray([max(1.0, _num(row.get("pixel_count"), _num(row.get("area_px"), 1.0))) for row in nodes], dtype=np.float32)
    d4rt_raw = np.asarray([max(0.0, _num(row.get("d4rt_witness_mass"))) for row in nodes], dtype=np.float32)
    neg_raw = np.asarray([max(0.0, _num(row.get("hard_negative_witness_mass"))) for row in nodes], dtype=np.float32)
    d4rt = np.log1p(d4rt_raw)
    neg = np.log1p(neg_raw)
    if d4rt.size and float(d4rt.max()) > 0:
        d4rt = d4rt / float(d4rt.max())
    if neg.size and float(neg.max()) > 0:
        neg = neg / float(neg.max())
    edge_dist = np.asarray([max(0.0, _num(row.get("source_edge_distance"))) for row in nodes], dtype=np.float32)
    boundary_token = np.asarray([1.0 if _bool(row.get("boundary_token")) else 0.0 for row in nodes], dtype=np.float32)
    near_edge = np.maximum(boundary_token * 0.35, 1.0 / (1.0 + edge_dist))
    background = np.asarray([max(0.0, _num(row.get("background_risk"))) for row in nodes], dtype=np.float32)
    broad = np.asarray([1.0 if _bool(row.get("broad_risk")) else 0.0 for row in nodes], dtype=np.float32)
    risk = np.clip(0.48 * neg + 0.22 * background + 0.18 * broad + 0.12 * near_edge, 0.0, 1.0)
    return {
        "area": area,
        "area_ratio": area / max(1.0, float(source_area)),
        "d4rt": d4rt.astype(np.float32),
        "negative": neg.astype(np.float32),
        "risk": risk.astype(np.float32),
        "near_edge": near_edge.astype(np.float32),
    }


def _score_for_spec(
    spec: dict[str, Any],
    obj_score: torch.Tensor,
    margin: torch.Tensor,
    d4rt: torch.Tensor,
    risk: torch.Tensor,
    repeated: bool,
) -> torch.Tensor:
    mode = str(spec["mode"])
    if mode == "d4rt":
        return 0.54 * d4rt + 0.30 * obj_score + 0.16 * torch.clamp(margin, min=0.0) - 0.25 * risk
    if mode == "radio":
        return 0.72 * obj_score + 0.22 * torch.clamp(margin, min=0.0) - 0.28 * risk + 0.05 * d4rt
    if mode == "consensus":
        return 0.44 * obj_score + 0.36 * d4rt + 0.20 * torch.clamp(margin, min=0.0) - 0.30 * risk
    if mode == "repeated":
        repeat_bonus = 0.12 if repeated else -0.18
        return 0.62 * obj_score + 0.18 * d4rt + 0.20 * torch.clamp(margin, min=0.0) - 0.30 * risk + repeat_bonus
    if mode == "separated":
        return 0.52 * obj_score + 0.45 * torch.clamp(margin, min=0.0) + 0.08 * d4rt - 0.28 * risk
    raise ValueError(mode)


def _select_indices(
    score: np.ndarray,
    margin: np.ndarray,
    risk: np.ndarray,
    area_ratio: np.ndarray,
    spec: dict[str, Any],
    repeated: bool,
) -> tuple[np.ndarray, str]:
    n = int(score.shape[0])
    if n == 0:
        return np.zeros(0, dtype=np.int64), "no_regions"
    max_regions = max(1, min(int(spec["max_regions"]), int(math.ceil(float(spec["top_fraction"]) * n))))
    order = np.argsort(score)[::-1]
    keep: list[int] = []
    running_area = 0.0
    for idx in order.tolist():
        if float(score[idx]) < float(spec["min_score"]):
            continue
        if float(risk[idx]) > float(spec["max_risk"]):
            continue
        if float(margin[idx]) < float(spec["min_margin"]):
            continue
        if running_area + float(area_ratio[idx]) > float(spec["max_area_ratio"]) and keep:
            continue
        keep.append(int(idx))
        running_area += float(area_ratio[idx])
        if len(keep) >= max_regions:
            break
    if keep:
        return np.asarray(keep, dtype=np.int64), "confirmed_core"
    if str(spec["mode"]) == "repeated" and not repeated:
        relaxed = [int(i) for i in order[: max_regions].tolist() if float(risk[i]) <= float(spec["max_risk"])]
        return np.asarray(relaxed, dtype=np.int64), "tentative_core"
    relaxed = [int(i) for i in order[: min(max_regions, 3)].tolist() if float(risk[i]) <= min(0.65, float(spec["max_risk"]) + 0.20)]
    return np.asarray(relaxed, dtype=np.int64), "tentative_core" if relaxed else "rejected_core"


def _expand_selected_indices(
    selected: np.ndarray,
    obj_score: np.ndarray,
    margin: np.ndarray,
    risk: np.ndarray,
    d4rt: np.ndarray,
    area_ratio: np.ndarray,
    spec: dict[str, Any],
    adjacency: dict[int, list[tuple[int, float, float]]],
) -> tuple[np.ndarray, dict[str, Any], set[int]]:
    selected_set = {int(i) for i in selected.tolist()}
    pre_area = float(sum(float(area_ratio[i]) for i in selected_set))
    stats: dict[str, Any] = {
        "area_repair_mode": spec.get("area_repair_mode", ""),
        "pre_expansion_core_region_count": len(selected_set),
        "pre_expansion_core_area_ratio": pre_area,
        "post_expansion_core_area_ratio": pre_area,
        "expansion_added_region_count": 0,
        "expansion_candidate_count": 0,
        "expansion_stop_reason": "not_needed",
    }
    if not selected_set or not adjacency:
        stats["expansion_stop_reason"] = "no_seed_or_no_adjacency"
        return selected, stats, set()

    max_area = float(spec["max_area_ratio"])
    target_area = min(max_area, float(spec.get("expansion_target_area_ratio", 0.022)))
    if pre_area >= target_area:
        return selected, stats, set()

    max_extra = max(0, int(spec.get("expansion_max_extra_regions", 0)))
    if max_extra <= 0:
        stats["expansion_stop_reason"] = "max_extra_zero"
        return selected, stats, set()

    seed_radio_mean = _mean([float(obj_score[i]) for i in selected_set])
    min_radio = max(float(spec.get("expansion_min_radio", 0.50)), seed_radio_mean - float(spec.get("expansion_radio_drop", 0.12)))
    min_margin = float(spec.get("expansion_min_margin", 0.006))
    min_edge_radio = float(spec.get("expansion_min_edge_radio", 0.82))
    max_risk = min(0.65, float(spec["max_risk"]) + float(spec.get("expansion_max_risk_delta", 0.08)))

    added: set[int] = set()
    current_area = pre_area
    stop_reason = "target_reached"
    while len(added) < max_extra and current_area < target_area:
        candidates: dict[int, tuple[float, float, float]] = {}
        for src in selected_set:
            for dst, edge_radio, edge_weight in adjacency.get(src, []):
                if dst in selected_set:
                    continue
                if float(edge_radio) < min_edge_radio:
                    continue
                if float(obj_score[dst]) < min_radio:
                    continue
                if float(margin[dst]) < min_margin:
                    continue
                if float(risk[dst]) > max_risk:
                    continue
                if current_area + float(area_ratio[dst]) > max_area:
                    continue
                candidate_score = (
                    0.48 * float(obj_score[dst])
                    + 0.24 * float(edge_radio)
                    + 0.12 * float(margin[dst])
                    + 0.08 * float(d4rt[dst])
                    + 0.08 * float(edge_weight)
                    - 0.22 * float(risk[dst])
                )
                previous = candidates.get(dst)
                if previous is None or candidate_score > previous[0]:
                    candidates[dst] = (candidate_score, float(edge_radio), float(edge_weight))
        stats["expansion_candidate_count"] = int(stats["expansion_candidate_count"]) + len(candidates)
        if not candidates:
            stop_reason = "no_radio_consistent_neighbor"
            break
        best_idx = max(candidates, key=lambda idx: candidates[idx][0])
        selected_set.add(int(best_idx))
        added.add(int(best_idx))
        current_area += float(area_ratio[best_idx])

    stats["post_expansion_core_area_ratio"] = current_area
    stats["expansion_added_region_count"] = len(added)
    stats["expansion_stop_reason"] = stop_reason if current_area >= target_area else "below_target_after_expansion"
    return np.asarray(sorted(selected_set), dtype=np.int64), stats, added


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    out = Path(args.output_root)
    if out.exists() and args.clean:
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    phase1 = _read_json(PHASE1 / "summary.json")
    if phase1.get("decision") != "PASS_V95_PHASE1_REGISTRY_READY":
        raise RuntimeError("v95 Phase1 must pass before Phase2")
    field_root = Path(args.field_root)
    shard_paths = sorted((field_root / "field_shards").glob("object_axis_unary_shard_*.npz"))
    if int(args.max_shards) > 0:
        shard_paths = shard_paths[: int(args.max_shards)]
    if not shard_paths:
        raise RuntimeError(f"no object-axis shards under {field_root / 'field_shards'}")

    source_meta, object_repeats = _load_phase1_sources()
    source_key_texts = _collect_shard_source_keys(shard_paths, 0, int(args.max_sources))
    selected_keys = {_source_key_parts(raw) for raw in source_key_texts}
    nodes_by_source = _load_region_nodes(PHASE1 / "region_node_rows.csv", selected_keys)
    edges_by_source = _load_region_edges(PHASE1 / "region_edge_rows.csv", selected_keys)
    specs = _variant_specs()
    created_at = _created_at()
    device = torch.device(str(args.device) if torch.cuda.is_available() and str(args.device).startswith("cuda") else "cpu")

    config_rows = [
        {
            "schema_version": "stream4d_v95_phase2_variant_config_v1",
            "phase_id": PHASE_ID,
            "run_id": RUN_ID,
            "created_at": created_at,
            **spec,
            "uses_gt_for_prediction": False,
            "uses_future": False,
        }
        for spec in specs
    ]
    core_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    casebook_rows: list[dict[str, Any]] = []
    object_core_sets: dict[tuple[str, str, str, int, int, str], dict[str, set[int]]] = defaultdict(dict)
    core_candidate_count = 0
    processed_source_count = 0
    processed_object_count = 0
    source_with_confirmed: set[tuple[str, str, int, int]] = set()
    score_backend_counts: Counter[str] = Counter()
    expansion_added_region_total = 0
    expanded_core_count = 0
    pre_expansion_area_ratios: list[float] = []

    source_limit = set(source_key_texts) if int(args.max_sources) > 0 else None
    for shard_index, shard_path in enumerate(shard_paths):
        with np.load(shard_path, allow_pickle=False) as data:
            source_keys = [str(value) for value in data["source_keys"].tolist()]
            object_keys = [str(value) for value in data["object_keys"].tolist()]
            object_source_index = data["object_source_index"].astype(np.int32)
            object_local_index = data["object_local_index"].astype(np.int32)
            region_source_index = data["region_source_index"].astype(np.int32)
            region_ids_all = [str(value) for value in data["region_ids"].tolist()]
            region_indices_all = data["region_indices"].astype(np.int32)
            unary_source_index = data["unary_source_index"].astype(np.int32)
            unary_object_local_index = data["unary_object_local_index"].astype(np.int32)
            unary_region_local_index = data["unary_region_local_index"].astype(np.int32)
            unary_cosine = data["unary_cosine"].astype(np.float32)

            for source_idx, raw_source_key in enumerate(source_keys):
                if source_limit is not None and raw_source_key not in source_limit:
                    continue
                key = _source_key_parts(raw_source_key)
                meta = source_meta.get(key)
                node_map = nodes_by_source.get(key, {})
                if not meta or not node_map:
                    failure_rows.append(
                        {
                            "schema_version": "stream4d_v95_phase2_failure_v1",
                            "phase_id": PHASE_ID,
                            "run_id": RUN_ID,
                            "source_key": raw_source_key,
                            "failure_type": "missing_phase1_source_or_region_nodes",
                            "uses_gt_for_prediction": False,
                            "uses_future": False,
                        }
                    )
                    continue
                region_mask = region_source_index == source_idx
                source_region_indices = region_indices_all[region_mask]
                source_region_ids = [region_ids_all[pos] for pos in np.nonzero(region_mask)[0].tolist()]
                nodes = [node_map.get(int(region_index)) for region_index in source_region_indices.tolist()]
                if any(node is None for node in nodes):
                    failure_rows.append(
                        {
                            "schema_version": "stream4d_v95_phase2_failure_v1",
                            "phase_id": PHASE_ID,
                            "run_id": RUN_ID,
                            "source_key": raw_source_key,
                            "failure_type": "region_index_missing_from_phase1_nodes",
                            "missing_count": sum(1 for node in nodes if node is None),
                            "uses_gt_for_prediction": False,
                            "uses_future": False,
                        }
                    )
                    continue
                nodes_typed = [node for node in nodes if node is not None]
                region_id_to_local = {str(region_id): idx for idx, region_id in enumerate(source_region_ids)}
                adjacency = _source_adjacency(edges_by_source.get(key, []), region_id_to_local)
                object_positions = np.nonzero(object_source_index == source_idx)[0]
                if object_positions.size == 0 or not nodes_typed:
                    continue
                object_pairs = sorted((int(object_local_index[pos]), object_keys[pos]) for pos in object_positions.tolist())
                k_objects = max(local_idx for local_idx, _ in object_pairs) + 1
                r_regions = len(nodes_typed)
                scores = np.full((k_objects, r_regions), -1.0, dtype=np.float32)
                source_unary = unary_source_index == source_idx
                for obj_idx, region_idx, value in zip(
                    unary_object_local_index[source_unary],
                    unary_region_local_index[source_unary],
                    unary_cosine[source_unary],
                    strict=False,
                ):
                    if 0 <= int(obj_idx) < k_objects and 0 <= int(region_idx) < r_regions:
                        scores[int(obj_idx), int(region_idx)] = float(value)
                if not np.any(scores >= 0.0):
                    continue

                feat = _node_features(nodes_typed, _num(meta.get("mask_area_px"), 1.0))
                score_t = torch.as_tensor(scores, dtype=torch.float32, device=device)
                d4rt_t = torch.as_tensor(feat["d4rt"], dtype=torch.float32, device=device)
                risk_t = torch.as_tensor(feat["risk"], dtype=torch.float32, device=device)
                top2 = torch.topk(score_t, k=min(2, score_t.shape[0]), dim=0).values
                second = top2[1] if top2.shape[0] > 1 else torch.zeros_like(top2[0])
                processed_source_count += 1
                score_backend_counts[str(device)] += 1

                for object_local_idx, object_id in object_pairs:
                    processed_object_count += 1
                    obj_score_t = score_t[int(object_local_idx)]
                    if score_t.shape[0] > 1:
                        other = torch.amax(torch.cat([score_t[: int(object_local_idx)], score_t[int(object_local_idx) + 1 :]], dim=0), dim=0)
                        margin_t = obj_score_t - other
                    else:
                        margin_t = obj_score_t - second
                    obj_score = obj_score_t.detach().cpu().numpy().astype(np.float32)
                    margin = margin_t.detach().cpu().numpy().astype(np.float32)
                    repeated = int(object_repeats.get(object_id, 0)) >= 2
                    for spec in specs:
                        core_candidate_count += 1
                        core_score_t = _score_for_spec(spec, obj_score_t, margin_t, d4rt_t, risk_t, repeated)
                        core_score = core_score_t.detach().cpu().numpy().astype(np.float32)
                        selected, state = _select_indices(core_score, margin, feat["risk"], feat["area_ratio"], spec, repeated)
                        seed_selected_set = {int(i) for i in selected.tolist()} if selected.size else set()
                        expansion_stats: dict[str, Any] = {
                            "area_repair_mode": spec.get("area_repair_mode", ""),
                            "pre_expansion_core_region_count": len(seed_selected_set),
                            "pre_expansion_core_area_ratio": float(sum(float(feat["area_ratio"][i]) for i in seed_selected_set)),
                            "post_expansion_core_area_ratio": float(sum(float(feat["area_ratio"][i]) for i in seed_selected_set)),
                            "expansion_added_region_count": 0,
                            "expansion_candidate_count": 0,
                            "expansion_stop_reason": "not_confirmed",
                        }
                        expanded_set: set[int] = set()
                        if state == "confirmed_core" and selected.size:
                            selected, expansion_stats, expanded_set = _expand_selected_indices(
                                selected,
                                obj_score,
                                margin,
                                feat["risk"],
                                feat["d4rt"],
                                feat["area_ratio"],
                                spec,
                                adjacency,
                            )
                            expansion_added_region_total += int(expansion_stats.get("expansion_added_region_count", 0) or 0)
                            if int(expansion_stats.get("expansion_added_region_count", 0) or 0) > 0:
                                expanded_core_count += 1
                            pre_expansion_area_ratios.append(float(expansion_stats.get("pre_expansion_core_area_ratio", 0.0) or 0.0))
                        selected_set = {int(i) for i in selected.tolist()} if selected.size else set()
                        if state == "confirmed_core" and selected_set:
                            source_with_confirmed.add(key)
                        object_core_sets[(spec["variant_id"], *key)][object_id] = selected_set if state == "confirmed_core" else set()
                        if not selected_set:
                            if len(casebook_rows) < int(args.casebook_limit):
                                casebook_rows.append(
                                    {
                                        "schema_version": "stream4d_v95_phase2_core_casebook_v1",
                                        "phase_id": PHASE_ID,
                                        "run_id": RUN_ID,
                                        "variant_id": spec["variant_id"],
                                        "scene_id": key[0],
                                        "window_id": key[1],
                                        "frame_id": key[2],
                                        "source_mask_id": key[3],
                                        "object_id": object_id,
                                        "case_type": "NO_SELECTED_CORE_REGION",
                                        "core_state": state,
                                        "top_core_score": float(np.max(core_score)) if core_score.size else "",
                                        "top_margin": float(np.max(margin)) if margin.size else "",
                                        "uses_gt_for_prediction": False,
                                        "uses_future": False,
                                    }
                                )
                            continue
                        order = selected[np.argsort(core_score[selected])[::-1]]
                        core_area = float(np.sum(feat["area"][order]))
                        source_area = max(1.0, _num(meta.get("mask_area_px"), 1.0))
                        radio_vals = [float(obj_score[idx]) for idx in order.tolist()]
                        d4rt_vals = [float(feat["d4rt"][idx]) for idx in order.tolist()]
                        risk_vals = [float(feat["risk"][idx]) for idx in order.tolist()]
                        margin_vals = [float(margin[idx]) for idx in order.tolist()]
                        for rank, idx in enumerate(order.tolist(), 1):
                            node = nodes_typed[int(idx)]
                            core_rows.append(
                                {
                                    "schema_version": "stream4d_v95_phase2_object_core_region_v1",
                                    "phase_id": PHASE_ID,
                                    "run_id": RUN_ID,
                                    "variant_id": spec["variant_id"],
                                    "scene_id": key[0],
                                    "window_id": key[1],
                                    "frame_id": key[2],
                                    "source_mask_id": key[3],
                                    "object_id": object_id,
                                    "region_id": source_region_ids[int(idx)],
                                    "region_index": int(source_region_indices[int(idx)]),
                                    "core_family": spec["core_family"],
                                    "core_score": float(core_score[int(idx)]),
                                    "core_rank": rank,
                                    "core_state": state,
                                    "core_seed_region": int(idx) in seed_selected_set,
                                    "expanded_from_neighbor": int(idx) in expanded_set,
                                    "core_area": float(feat["area"][int(idx)]),
                                    "source_area": source_area,
                                    "core_area_ratio": float(feat["area"][int(idx)] / source_area),
                                    "pre_expansion_core_area_ratio": float(expansion_stats.get("pre_expansion_core_area_ratio", 0.0) or 0.0),
                                    "post_expansion_core_area_ratio": float(expansion_stats.get("post_expansion_core_area_ratio", 0.0) or 0.0),
                                    "radio_consistency": float(obj_score[int(idx)]),
                                    "D4RT_witness_score": float(feat["d4rt"][int(idx)]),
                                    "D4RT_negative_score": float(feat["negative"][int(idx)]),
                                    "risk_score": float(feat["risk"][int(idx)]),
                                    "conflict_score": float(max(0.0, -margin[int(idx)])),
                                    "temporal_support_proxy": int(object_repeats.get(object_id, 0)),
                                    "selected_as_core": state == "confirmed_core",
                                    "reason_selected": "radio_consistent_neighbor_expansion" if int(idx) in expanded_set else f"{spec['mode']}_pre_registered_thresholds",
                                    "uses_gt_for_prediction": False,
                                    "uses_future": False,
                                }
                            )
                        summary_rows.append(
                            {
                                "schema_version": "stream4d_v95_phase2_object_core_summary_v1",
                                "phase_id": PHASE_ID,
                                "run_id": RUN_ID,
                                "variant_id": spec["variant_id"],
                                "scene_id": key[0],
                                "window_id": key[1],
                                "frame_id": key[2],
                                "source_mask_id": key[3],
                                "object_id": object_id,
                                "core_family": spec["core_family"],
                                "core_state": state,
                                "core_region_count": len(order),
                                "pre_expansion_core_region_count": int(expansion_stats.get("pre_expansion_core_region_count", len(seed_selected_set)) or 0),
                                "expansion_added_region_count": int(expansion_stats.get("expansion_added_region_count", 0) or 0),
                                "core_area_ratio": core_area / source_area,
                                "pre_expansion_core_area_ratio": float(expansion_stats.get("pre_expansion_core_area_ratio", core_area / source_area) or 0.0),
                                "post_expansion_core_area_ratio": float(expansion_stats.get("post_expansion_core_area_ratio", core_area / source_area) or 0.0),
                                "D4RT_support_mass_mean": _mean(d4rt_vals),
                                "RADIO_consistency_mean": _mean(radio_vals),
                                "risk_score_mean": _mean(risk_vals),
                                "conflict_score_mean": _mean([max(0.0, -v) for v in margin_vals]),
                                "core_object_margin_mean": _mean(margin_vals),
                                "core_entropy_mean": 0.0,
                                "area_repair_mode": expansion_stats.get("area_repair_mode", ""),
                                "expansion_candidate_count": int(expansion_stats.get("expansion_candidate_count", 0) or 0),
                                "expansion_stop_reason": expansion_stats.get("expansion_stop_reason", ""),
                                "uses_gt_for_prediction": False,
                                "uses_future": False,
                            }
                        )
                if int(args.progress_every_sources) > 0 and processed_source_count % int(args.progress_every_sources) == 0:
                    print(
                        json.dumps(
                            {
                                "phase": PHASE_ID,
                                "processed_source_count": processed_source_count,
                                "processed_object_count": processed_object_count,
                                "core_region_rows": len(core_rows),
                                "elapsed_sec": time.time() - started,
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )

    confirmed_summaries = [row for row in summary_rows if row.get("core_state") == "confirmed_core"]
    tentative_summaries = [row for row in summary_rows if row.get("core_state") == "tentative_core"]
    quarantine_summaries = [row for row in summary_rows if row.get("core_state") == "quarantine_core"]
    core_area_ratios = [_num(row.get("core_area_ratio")) for row in confirmed_summaries]
    conflict_scores = [_num(row.get("conflict_score_mean")) for row in confirmed_summaries]
    overlap_rates: list[float] = []
    for source_key, object_sets in object_core_sets.items():
        region_owner_count: Counter[int] = Counter()
        total = 0
        for selected_set in object_sets.values():
            total += len(selected_set)
            for idx in selected_set:
                region_owner_count[int(idx)] += 1
        overlap = sum(count - 1 for count in region_owner_count.values() if count > 1)
        overlap_rates.append(float(overlap / max(1, total)))
    core_overlap_rate = _mean(overlap_rates)
    source_with_confirmed_rate = len(source_with_confirmed) / max(1, processed_source_count)
    phase2_pass = bool(
        len(confirmed_summaries) >= 100
        and source_with_confirmed_rate >= 0.20
        and 0.02 <= _mean(core_area_ratios) <= 0.35
        and core_overlap_rate <= 0.05
        and _mean(conflict_scores) <= 0.10
    )
    blocker = ""
    if not phase2_pass:
        if len(confirmed_summaries) < 100 or source_with_confirmed_rate < 0.20:
            blocker = "OBJECT_CORE_TOO_TENTATIVE"
        elif core_overlap_rate > 0.05 or _mean(conflict_scores) > 0.10:
            blocker = "OBJECT_CORE_NOT_CLEAN"
        else:
            blocker = "OBJECT_CORE_DISCOVERY_BLOCKER"

    gate_rows = [
        {
            "schema_version": "stream4d_v95_phase2_gate_v1",
            "phase_id": PHASE_ID,
            "run_id": RUN_ID,
            "criterion": "confirmed_core_count_ge_100",
            "pass": len(confirmed_summaries) >= 100,
            "observed": len(confirmed_summaries),
            "required": 100,
            "uses_gt_for_prediction": False,
            "uses_future": False,
        },
        {
            "schema_version": "stream4d_v95_phase2_gate_v1",
            "phase_id": PHASE_ID,
            "run_id": RUN_ID,
            "criterion": "source_with_at_least_one_confirmed_core_rate_ge_0p20",
            "pass": source_with_confirmed_rate >= 0.20,
            "observed": source_with_confirmed_rate,
            "required": 0.20,
            "uses_gt_for_prediction": False,
            "uses_future": False,
        },
        {
            "schema_version": "stream4d_v95_phase2_gate_v1",
            "phase_id": PHASE_ID,
            "run_id": RUN_ID,
            "criterion": "core_area_ratio_mean_between_0p02_0p35",
            "pass": 0.02 <= _mean(core_area_ratios) <= 0.35,
            "observed": _mean(core_area_ratios),
            "required": "0.02..0.35",
            "uses_gt_for_prediction": False,
            "uses_future": False,
        },
        {
            "schema_version": "stream4d_v95_phase2_gate_v1",
            "phase_id": PHASE_ID,
            "run_id": RUN_ID,
            "criterion": "core_overlap_between_objects_rate_le_0p05",
            "pass": core_overlap_rate <= 0.05,
            "observed": core_overlap_rate,
            "required": 0.05,
            "uses_gt_for_prediction": False,
            "uses_future": False,
        },
        {
            "schema_version": "stream4d_v95_phase2_gate_v1",
            "phase_id": PHASE_ID,
            "run_id": RUN_ID,
            "criterion": "core_conflict_score_mean_le_0p10",
            "pass": _mean(conflict_scores) <= 0.10,
            "observed": _mean(conflict_scores),
            "required": 0.10,
            "uses_gt_for_prediction": False,
            "uses_future": False,
        },
    ]
    summary = {
        "schema": "stream4d_v95_phase2_object_core_discovery_summary_v1",
        "phase_id": PHASE_ID,
        "run_id": RUN_ID,
        "created_at": created_at,
        "decision": "PASS_V95_PHASE2_OBJECT_CORE_READY" if phase2_pass else "NO_GO_V95_PHASE2_OBJECT_CORE_DISCOVERY",
        "blocker": blocker,
        "field_root": _rel(field_root),
        "field_shard_count": len(shard_paths),
        "processed_source_count": processed_source_count,
        "processed_object_count": processed_object_count,
        "core_candidate_count": core_candidate_count,
        "confirmed_core_count": len(confirmed_summaries),
        "tentative_core_count": len(tentative_summaries),
        "quarantine_core_count": len(quarantine_summaries),
        "core_region_count_mean": _mean([_num(row.get("core_region_count")) for row in confirmed_summaries]),
        "core_area_ratio_mean": _mean(core_area_ratios),
        "pre_expansion_core_area_ratio_mean": _mean(pre_expansion_area_ratios),
        "core_area_ratio_p10": _percentile(core_area_ratios, 10),
        "core_area_ratio_p90": _percentile(core_area_ratios, 90),
        "area_repair_mode": "radio_consistent_neighbor_expansion",
        "expanded_core_count": expanded_core_count,
        "expansion_added_region_count": expansion_added_region_total,
        "D4RT_support_mass_mean": _mean([_num(row.get("D4RT_support_mass_mean")) for row in confirmed_summaries]),
        "RADIO_consistency_mean": _mean([_num(row.get("RADIO_consistency_mean")) for row in confirmed_summaries]),
        "risk_score_mean": _mean([_num(row.get("risk_score_mean")) for row in confirmed_summaries]),
        "conflict_score_mean": _mean(conflict_scores),
        "core_to_source_area_ratio": _mean(core_area_ratios),
        "core_overlap_between_objects_rate": core_overlap_rate,
        "core_object_margin_mean": _mean([_num(row.get("core_object_margin_mean")) for row in confirmed_summaries]),
        "core_entropy_mean": _mean([_num(row.get("core_entropy_mean")) for row in confirmed_summaries]),
        "confirmed_core_rate": len(confirmed_summaries) / max(1, core_candidate_count),
        "source_with_at_least_one_confirmed_core_rate": source_with_confirmed_rate,
        "diagnostic_gt_metrics_available": False,
        "core_precision_diagnostic": "",
        "core_recall_diagnostic": "",
        "core_same_gt_purity_diagnostic": "",
        "core_wrong_gt_rate_diagnostic": "",
        "same_semantic_hard_negative_AUC_diagnostic": "",
        "score_backend_counts": dict(score_backend_counts),
        "uses_gt_for_prediction_count": 0,
        "uses_future_count": 0,
        "uses_gt_for_prediction": False,
        "uses_future": False,
        "duration_sec": time.time() - started,
        "row_counts": {
            "variant_config_rows": len(config_rows),
            "object_core_region_rows": len(core_rows),
            "object_core_summary_rows": len(summary_rows),
            "object_core_failure_rows": len(failure_rows),
            "object_core_casebook_rows": len(casebook_rows),
            "variant_gate_rows": len(gate_rows),
        },
    }
    _write_csv(out / "variant_config_rows.csv", config_rows)
    _write_csv(out / "object_core_region_rows.csv", core_rows)
    _write_csv(out / "object_core_summary_rows.csv", summary_rows)
    _write_csv(out / "object_core_failure_rows.csv", failure_rows)
    _write_csv(out / "object_core_casebook_rows.csv", casebook_rows)
    _write_csv(out / "variant_gate_rows.csv", gate_rows)
    _write_json(out / "summary.json", summary)
    outputs = [
        out / "variant_config_rows.csv",
        out / "object_core_region_rows.csv",
        out / "object_core_summary_rows.csv",
        out / "object_core_failure_rows.csv",
        out / "object_core_casebook_rows.csv",
        out / "variant_gate_rows.csv",
        out / "summary.json",
    ]
    _write_json(out / "SHA256SUMS.json", {_rel(path): _sha256(path) for path in outputs if path.exists()})
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True), flush=True)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--field-root", default=str(DEFAULT_FIELD_ROOT))
    parser.add_argument("--output-root", default=str(OUT))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-shards", type=int, default=0)
    parser.add_argument("--max-sources", type=int, default=0)
    parser.add_argument("--progress-every-sources", type=int, default=256)
    parser.add_argument("--casebook-limit", type=int, default=500)
    parser.add_argument("--clean", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
