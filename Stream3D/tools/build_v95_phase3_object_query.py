#!/usr/bin/env python3
"""Build v95 object queries from confirmed object cores."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
PHASE_ID = "v95_phase3_object_query"
RUN_ID = "v95_phase3_object_query"
OUT = ROOT / "outputs/audit/v95_phase3_object_query"
PHASE1 = ROOT / "outputs/audit/v95_phase1_physical_source_registry"
DEFAULT_CORE_ROOT = ROOT / "outputs/audit/v95_phase2_object_core_discovery_repair1"
DEFAULT_FIELD_ROOT = ROOT / "outputs/audit/v94_phase7c_object_axis_field_full_dev_combined"

QUERY_FAMILIES = [
    ("Q0_mask_average_prototype_control", "mask_average", False),
    ("Q1_core_mean", "mean", True),
    ("Q2_core_medoid", "medoid", True),
    ("Q3_trimmed_core_mean", "trimmed_mean", True),
    ("Q4_D4RT_weighted_core_mean", "d4rt_weighted_mean", True),
    ("Q5_RADIO_consistency_weighted_medoid", "radio_weighted_medoid", True),
    ("Q6_temporal_repeat_weighted_core_query", "temporal_weighted_mean", True),
]
FEATURE_NAMES = [
    "object_axis_unary",
    "positive_object_margin",
    "d4rt_witness",
    "non_negative_witness",
    "low_risk",
    "interior_context",
    "source_mean_cosine",
    "area_signal",
    "temporal_repeat_signal",
]
QUALITY_LOGIT_SCALE = 8.0


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
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _num(value: Any, default: float = 0.0) -> float:
    if value is None or value == "":
        return default
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(out):
        return default
    return out


def _int(value: Any, default: int = -1) -> int:
    if value is None or value == "":
        return default
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _mean(values: list[float]) -> float:
    clean = [float(v) for v in values if math.isfinite(float(v))]
    return float(sum(clean) / len(clean)) if clean else 0.0


def _percentile(values: list[float], q: float) -> float:
    clean = [float(v) for v in values if math.isfinite(float(v))]
    return float(np.percentile(np.asarray(clean, dtype=np.float64), q)) if clean else 0.0


def _source_key(row: dict[str, Any]) -> tuple[str, str, int, int]:
    return (str(row["scene_id"]), str(row["window_id"]), _int(row["frame_id"]), _int(row["source_mask_id"]))


def _source_key_parts(raw: str) -> tuple[str, str, int, int]:
    scene, window, frame_raw, mask_raw = str(raw).split("|")
    return scene, window, int(frame_raw), int(mask_raw)


def _source_key_text(key: tuple[str, str, int, int]) -> str:
    return f"{key[0]}|{key[1]}|{key[2]}|{key[3]}"


def _core_key(row: dict[str, Any]) -> tuple[str, str, str, int, int, str]:
    key = _source_key(row)
    return (str(row["variant_id"]), key[0], key[1], key[2], key[3], str(row["object_id"]))


def _core_selection_score(row: dict[str, Any]) -> float:
    area = _num(row.get("core_area_ratio"))
    area_bonus = max(0.0, 1.0 - abs(area - 0.026) / 0.026)
    return (
        0.36 * _num(row.get("RADIO_consistency_mean"))
        + 0.22 * _num(row.get("core_object_margin_mean"))
        + 0.18 * _num(row.get("D4RT_support_mass_mean"))
        + 0.10 * area_bonus
        - 0.20 * _num(row.get("risk_score_mean"))
        - 0.20 * _num(row.get("conflict_score_mean"))
    )


def _load_phase1_sources() -> dict[tuple[str, str, int, int], dict[str, str]]:
    sources: dict[tuple[str, str, int, int], dict[str, str]] = {}
    with (PHASE1 / "source_container_rows.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if _bool(row.get("uses_gt_for_prediction")) or _bool(row.get("uses_future")):
                continue
            sources[_source_key(row)] = dict(row)
    return sources


def _load_region_nodes(path: Path, selected_keys: set[tuple[str, str, int, int]]) -> dict[tuple[str, str, int, int], dict[int, dict[str, str]]]:
    nodes: dict[tuple[str, str, int, int], dict[int, dict[str, str]]] = defaultdict(dict)
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = _source_key(row)
            if key not in selected_keys:
                continue
            if _bool(row.get("uses_gt_for_prediction")) or _bool(row.get("uses_future")):
                continue
            nodes[key][_int(row.get("region_index"))] = dict(row)
    return dict(nodes)


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
    source_mean = np.asarray([np.clip(_num(row.get("source_mean_cosine")), 0.0, 1.0) for row in nodes], dtype=np.float32)
    return {
        "area": area,
        "area_ratio": area / max(1.0, float(source_area)),
        "d4rt": d4rt.astype(np.float32),
        "negative": neg.astype(np.float32),
        "risk": risk.astype(np.float32),
        "near_edge": near_edge.astype(np.float32),
        "source_mean": source_mean,
    }


def _load_selected_cores(core_root: Path, max_sources: int = 0) -> tuple[
    dict[tuple[str, str, int, int], list[dict[str, Any]]],
    dict[tuple[str, str, str, int, int, str], list[int]],
    dict[str, Any],
]:
    summary_path = core_root / "object_core_summary_rows.csv"
    region_path = core_root / "object_core_region_rows.csv"
    phase2_summary = json.loads((core_root / "summary.json").read_text(encoding="utf-8"))
    confirmed: list[dict[str, Any]] = []
    with summary_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("core_state") != "confirmed_core":
                continue
            if _bool(row.get("uses_gt_for_prediction")) or _bool(row.get("uses_future")):
                continue
            row = dict(row)
            row["core_selection_score"] = _core_selection_score(row)
            confirmed.append(row)

    source_order: list[tuple[str, str, int, int]] = []
    seen_sources: set[tuple[str, str, int, int]] = set()
    for row in confirmed:
        key = _source_key(row)
        if key not in seen_sources:
            seen_sources.add(key)
            source_order.append(key)
    if max_sources > 0:
        allowed = set(source_order[:max_sources])
        confirmed = [row for row in confirmed if _source_key(row) in allowed]

    best_by_object: dict[tuple[str, str, int, int, str], dict[str, Any]] = {}
    for row in confirmed:
        source = _source_key(row)
        key = (*source, str(row["object_id"]))
        previous = best_by_object.get(key)
        if previous is None or float(row["core_selection_score"]) > float(previous["core_selection_score"]):
            best_by_object[key] = row

    selected_core_keys = {_core_key(row) for row in best_by_object.values()}
    selected_regions: dict[tuple[str, str, str, int, int, str], list[int]] = defaultdict(list)
    seed_regions: dict[tuple[str, str, str, int, int, str], list[int]] = defaultdict(list)
    with region_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("core_state") != "confirmed_core":
                continue
            if not _bool(row.get("selected_as_core")):
                continue
            if _bool(row.get("uses_gt_for_prediction")) or _bool(row.get("uses_future")):
                continue
            key = _core_key(row)
            if key not in selected_core_keys:
                continue
            selected_regions[key].append(_int(row.get("region_index")))
            if _bool(row.get("core_seed_region")):
                seed_regions[key].append(_int(row.get("region_index")))

    selected_by_source: dict[tuple[str, str, int, int], list[dict[str, Any]]] = defaultdict(list)
    dropped_no_region = 0
    for row in best_by_object.values():
        key = _core_key(row)
        regions = sorted(set(selected_regions.get(key, [])))
        if not regions:
            dropped_no_region += 1
            continue
        row = dict(row)
        row["selected_core_region_indices"] = regions
        row["query_core_region_indices"] = sorted(set(seed_regions.get(key, []))) or regions
        row["query_core_policy"] = "seed_core_only_if_available_else_confirmed_core"
        selected_by_source[_source_key(row)].append(row)

    meta = {
        "raw_phase2_confirmed_core_count": len(confirmed),
        "selected_confirmed_core_count": sum(len(v) for v in selected_by_source.values()),
        "selected_source_count": len(selected_by_source),
        "dropped_selected_core_without_region_count": dropped_no_region,
        "phase2_decision": phase2_summary.get("decision"),
        "phase2_core_area_ratio_mean": phase2_summary.get("core_area_ratio_mean"),
        "phase2_core_overlap_between_objects_rate": phase2_summary.get("core_overlap_between_objects_rate"),
    }
    return dict(selected_by_source), dict(selected_regions), meta


def _region_feature_matrix(
    scores: np.ndarray,
    object_local_idx: int,
    feat: dict[str, np.ndarray],
    repeat_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    obj_score = np.clip(scores[int(object_local_idx)], 0.0, 1.0)
    if scores.shape[0] > 1:
        other = np.max(np.concatenate([scores[: int(object_local_idx)], scores[int(object_local_idx) + 1 :]], axis=0), axis=0)
    else:
        other = np.zeros_like(obj_score)
    margin = obj_score - other
    repeat_signal = np.full_like(obj_score, min(1.0, float(repeat_count) / 8.0), dtype=np.float32)
    area_signal = np.clip(feat["area_ratio"] * 24.0, 0.0, 1.0)
    matrix = np.stack(
        [
            obj_score,
            np.clip(margin, 0.0, 1.0),
            feat["d4rt"],
            1.0 - feat["negative"],
            1.0 - feat["risk"],
            1.0 - np.clip(feat["near_edge"], 0.0, 1.0),
            feat["source_mean"],
            area_signal,
            repeat_signal,
        ],
        axis=1,
    ).astype(np.float32)
    return matrix, obj_score.astype(np.float32), margin.astype(np.float32), other.astype(np.float32)


def _cosine_scores(matrix: np.ndarray, query: np.ndarray, device: torch.device) -> np.ndarray:
    x = torch.as_tensor(matrix, dtype=torch.float32, device=device)
    q = torch.as_tensor(query, dtype=torch.float32, device=device)
    x = torch.nn.functional.normalize(x, p=2, dim=1, eps=1e-8)
    q = torch.nn.functional.normalize(q, p=2, dim=0, eps=1e-8)
    return torch.mv(x, q).detach().cpu().numpy().astype(np.float32)


def _entropy_from_vector(vec: np.ndarray) -> float:
    mag = np.abs(vec.astype(np.float64)) + 1e-12
    prob = mag / float(np.sum(mag))
    return float(-np.sum(prob * np.log(prob)) / math.log(max(2, len(prob))))


def _query_vectors_for_core(
    matrix: np.ndarray,
    core_local_indices: list[int],
    obj_score: np.ndarray,
    margin: np.ndarray,
    feat: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    core = np.asarray(core_local_indices, dtype=np.int64)
    if core.size == 0:
        raise ValueError("empty core")
    all_mean = np.average(matrix, axis=0, weights=np.maximum(feat["area"], 1.0))
    centered = matrix - all_mean
    core_matrix = centered[core]
    mean = np.mean(core_matrix, axis=0)
    norm_core = core_matrix / np.maximum(np.linalg.norm(core_matrix, axis=1, keepdims=True), 1e-8)
    norm_mean = mean / max(float(np.linalg.norm(mean)), 1e-8)
    medoid = core_matrix[int(np.argmax(norm_core @ norm_mean))]
    quality = 0.46 * obj_score[core] + 0.24 * np.clip(margin[core], 0.0, 1.0) + 0.18 * feat["d4rt"][core] - 0.20 * feat["risk"][core]
    keep_n = max(1, int(math.ceil(0.80 * len(core))))
    keep = core[np.argsort(quality)[::-1][:keep_n]]
    trimmed = np.mean(centered[keep], axis=0)
    d4rt_w = feat["d4rt"][core] + 0.05
    d4rt_weighted = np.average(core_matrix, axis=0, weights=d4rt_w)
    radio_w = obj_score[core] + 0.05
    radio_mean = np.average(core_matrix, axis=0, weights=radio_w)
    norm_radio = radio_mean / max(float(np.linalg.norm(radio_mean)), 1e-8)
    radio_medoid = core_matrix[int(np.argmax(norm_core @ norm_radio))]
    temporal_w = core_matrix[:, -1] + 1.0
    temporal_weighted = np.average(core_matrix, axis=0, weights=temporal_w)
    return {
        "Q0_mask_average_prototype_control": all_mean.astype(np.float32),
        "Q1_core_mean": mean.astype(np.float32),
        "Q2_core_medoid": medoid.astype(np.float32),
        "Q3_trimmed_core_mean": trimmed.astype(np.float32),
        "Q4_D4RT_weighted_core_mean": d4rt_weighted.astype(np.float32),
        "Q5_RADIO_consistency_weighted_medoid": radio_medoid.astype(np.float32),
        "Q6_temporal_repeat_weighted_core_query": temporal_weighted.astype(np.float32),
    }


@dataclass
class QualityAcc:
    query_count: int = 0
    source_count: set[tuple[str, str, int, int]] = field(default_factory=set)
    norm_values: list[float] = field(default_factory=list)
    entropy_values: list[float] = field(default_factory=list)
    query_margin_values: list[float] = field(default_factory=list)
    unary_margin_values: list[float] = field(default_factory=list)
    unary_entropy_values: list[float] = field(default_factory=list)
    background_margin_values: list[float] = field(default_factory=list)
    competing_margin_values: list[float] = field(default_factory=list)
    query_vs_mask_values: list[float] = field(default_factory=list)
    risk_weighted_unary_values: list[float] = field(default_factory=list)

    def row(self, family: str, confirmed_core_count: int, selected_for_expansion: bool) -> dict[str, Any]:
        return {
            "schema_version": "stream4d_v95_phase3_query_quality_v1",
            "phase_id": PHASE_ID,
            "run_id": RUN_ID,
            "query_family": family,
            "selected_for_expansion": selected_for_expansion,
            "query_count": self.query_count,
            "confirmed_core_count": confirmed_core_count,
            "query_coverage_rate": self.query_count / max(1, confirmed_core_count),
            "query_norm_mean": _mean(self.norm_values),
            "query_entropy_mean": _mean(self.entropy_values),
            "query_margin_proxy_mean": _mean(self.query_margin_values),
            "unary_margin_mean": _mean(self.unary_margin_values),
            "unary_entropy_mean": _mean(self.unary_entropy_values),
            "background_unary_margin": _mean(self.background_margin_values),
            "competing_object_margin_mean": _mean(self.competing_margin_values),
            "object_query_vs_mask_average_cosine_mean": _mean(self.query_vs_mask_values),
            "risk_weighted_unary_mean": _mean(self.risk_weighted_unary_values),
            "uses_gt_for_prediction": False,
            "uses_future": False,
        }


def _softmax_entropy(scores: np.ndarray) -> float:
    if scores.size == 0 or scores.shape[0] <= 1:
        return 0.0
    shifted = scores - np.max(scores, axis=0, keepdims=True)
    exp = np.exp(np.clip(shifted, -40.0, 40.0))
    prob = exp / np.maximum(np.sum(exp, axis=0, keepdims=True), 1e-12)
    ent = -np.sum(prob * np.log(np.maximum(prob, 1e-12)), axis=0) / math.log(scores.shape[0])
    return float(np.mean(ent))


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    out = Path(args.output_root)
    if out.exists() and args.clean:
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    core_root = Path(args.core_root)
    field_root = Path(args.field_root)
    phase2 = json.loads((core_root / "summary.json").read_text(encoding="utf-8"))
    if phase2.get("decision") != "PASS_V95_PHASE2_OBJECT_CORE_READY":
        raise RuntimeError("v95 Phase2 must pass before Phase3")
    phase1_sources = _load_phase1_sources()
    selected_by_source, _selected_regions, core_meta = _load_selected_cores(core_root, int(args.max_sources))
    selected_source_keys = set(selected_by_source)
    nodes_by_source = _load_region_nodes(PHASE1 / "region_node_rows.csv", selected_source_keys)
    shard_paths = sorted((field_root / "field_shards").glob("object_axis_unary_shard_*.npz"))
    if int(args.max_shards) > 0:
        shard_paths = shard_paths[: int(args.max_shards)]
    if not shard_paths:
        raise RuntimeError(f"no object-axis shards under {field_root / 'field_shards'}")

    device = torch.device(str(args.device) if torch.cuda.is_available() and str(args.device).startswith("cuda") else "cpu")
    created_at = _created_at()
    feature_dim = len(FEATURE_NAMES)
    confirmed_core_count = int(core_meta["selected_confirmed_core_count"])
    query_rows: list[dict[str, Any]] = []
    vector_index_rows: list[dict[str, Any]] = []
    vectors: list[np.ndarray] = []
    query_ids: list[str] = []
    quality: dict[str, QualityAcc] = {family: QualityAcc() for family, _, _ in QUERY_FAMILIES}
    casebook_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    source_processed = 0
    selected_core_processed = 0
    score_backend_counts: Counter[str] = Counter()
    query_ref_by_key: dict[tuple[str, str, str, int, int, str, str], str] = {}

    for shard_path in shard_paths:
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
                key = _source_key_parts(raw_source_key)
                source_cores = selected_by_source.get(key, [])
                if not source_cores:
                    continue
                meta = phase1_sources.get(key)
                node_map = nodes_by_source.get(key, {})
                if not meta or not node_map:
                    failure_rows.append(
                        {
                            "schema_version": "stream4d_v95_phase3_failure_v1",
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
                            "schema_version": "stream4d_v95_phase3_failure_v1",
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
                object_positions = np.nonzero(object_source_index == source_idx)[0]
                if object_positions.size == 0:
                    continue
                object_pairs = sorted((int(object_local_index[pos]), object_keys[pos]) for pos in object_positions.tolist())
                object_id_to_local = {object_id: local_idx for local_idx, object_id in object_pairs}
                k_objects = max(object_id_to_local.values()) + 1
                r_regions = len(source_region_indices)
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
                feat = _node_features([node for node in nodes if node is not None], _num(meta.get("mask_area_px"), 1.0))
                region_index_to_local = {int(region_index): idx for idx, region_index in enumerate(source_region_indices.tolist())}
                family_source_records: dict[str, list[dict[str, Any]]] = defaultdict(list)
                source_processed += 1
                score_backend_counts[str(device)] += 1

                for core_row in source_cores:
                    object_id = str(core_row["object_id"])
                    if object_id not in object_id_to_local:
                        if len(casebook_rows) < int(args.casebook_limit):
                            casebook_rows.append(
                                {
                                    "schema_version": "stream4d_v95_phase3_casebook_v1",
                                    "phase_id": PHASE_ID,
                                    "run_id": RUN_ID,
                                    "case_type": "selected_core_object_missing_from_unary_shard",
                                    "source_key": raw_source_key,
                                    "object_id": object_id,
                                    "uses_gt_for_prediction": False,
                                    "uses_future": False,
                                }
                            )
                        continue
                    local_core = [region_index_to_local[idx] for idx in core_row["query_core_region_indices"] if idx in region_index_to_local]
                    if not local_core:
                        continue
                    object_local_idx = object_id_to_local[object_id]
                    matrix, obj_score, margin, _other = _region_feature_matrix(
                        scores,
                        int(object_local_idx),
                        feat,
                        _int(core_row.get("temporal_support_proxy"), 0),
                    )
                    query_vectors = _query_vectors_for_core(matrix, local_core, obj_score, margin, feat)
                    mask_query = query_vectors["Q0_mask_average_prototype_control"]
                    for family, agg_mode, is_method in QUERY_FAMILIES:
                        q = query_vectors[family]
                        vector_index = len(vectors)
                        query_ref = f"{RUN_ID}:q{vector_index:07d}"
                        query_id = (
                            f"{core_row['variant_id']}|{family}|{key[0]}|{key[1]}|{key[2]}|"
                            f"{key[3]}|{object_id}"
                        )
                        query_ids.append(query_id)
                        vectors.append(q.astype(np.float32))
                        query_ref_by_key[(str(core_row["variant_id"]), key[0], key[1], key[2], key[3], object_id, family)] = query_ref
                        q_norm = float(np.linalg.norm(q))
                        query_rows.append(
                            {
                                "schema_version": "stream4d_v95_phase3_object_query_v1",
                                "phase_id": PHASE_ID,
                                "run_id": RUN_ID,
                                "created_at": created_at,
                                "scene_id": key[0],
                                "window_id": key[1],
                                "frame_id": key[2],
                                "source_mask_id": key[3],
                                "object_id": object_id,
                                "core_variant_id": core_row["variant_id"],
                                "query_family": family,
                                "core_region_count": len(local_core),
                                "phase2_core_region_count": len(core_row["selected_core_region_indices"]),
                                "query_core_policy": core_row["query_core_policy"],
                                "core_area_ratio": core_row.get("core_area_ratio", ""),
                                "radio_query_norm": q_norm,
                                "query_entropy": _entropy_from_vector(q),
                                "query_margin_proxy": _mean([float(margin[i]) for i in local_core]),
                                "query_risk_mean": _mean([float(feat["risk"][i]) for i in local_core]),
                                "query_conflict_mean": _mean([float(max(0.0, -margin[i])) for i in local_core]),
                                "robust_agg_mode": agg_mode,
                                "query_centering_mode": "raw_mask_average_control" if family == "Q0_mask_average_prototype_control" else "source_centered_residual",
                                "has_confirmed_core": True,
                                "query_vector_ref": query_ref,
                                "query_feature_dim": feature_dim,
                                "is_method_query": is_method,
                                "uses_gt_for_prediction": False,
                                "uses_future": False,
                            }
                        )
                        vector_index_rows.append(
                            {
                                "schema_version": "stream4d_v95_phase3_query_vector_index_v1",
                                "phase_id": PHASE_ID,
                                "run_id": RUN_ID,
                                "query_vector_ref": query_ref,
                                "npz_key": "query_vectors",
                                "vector_index": vector_index,
                                "query_id": query_id,
                                "query_family": family,
                                "feature_dim": feature_dim,
                                "feature_names": "|".join(FEATURE_NAMES),
                                "sha256": "",
                                "uses_gt_for_prediction": False,
                                "uses_future": False,
                            }
                        )
                        acc = quality[family]
                        acc.query_count += 1
                        acc.source_count.add(key)
                        acc.norm_values.append(q_norm)
                        acc.entropy_values.append(_entropy_from_vector(q))
                        acc.query_margin_values.append(_mean([float(margin[i]) for i in local_core]))
                        mask_cos = float(np.dot(q, mask_query) / max(1e-8, float(np.linalg.norm(q)) * float(np.linalg.norm(mask_query))))
                        acc.query_vs_mask_values.append(mask_cos)
                        family_source_records[family].append(
                            {
                                "object_id": object_id,
                                "core_local_indices": local_core,
                                "query": q,
                                "matrix": matrix if family == "Q0_mask_average_prototype_control" else matrix - mask_query,
                                "risk": feat["risk"],
                                "negative": feat["negative"],
                                "d4rt": feat["d4rt"],
                                "margin": margin,
                            }
                        )
                    selected_core_processed += 1

                for family, records in family_source_records.items():
                    if not records:
                        continue
                    score_rows = []
                    for record in records:
                        proto = _cosine_scores(record["matrix"], record["query"], device)
                        score_rows.append(
                            0.70 * proto
                            + 0.16 * record["d4rt"]
                            + 0.08 * np.clip(record["margin"], 0.0, 1.0)
                            - 0.12 * record["risk"]
                            - 0.06 * record["negative"]
                        )
                    score_matrix = np.stack(score_rows, axis=0)
                    for rec_idx, record in enumerate(records):
                        core_idx = np.asarray(record["core_local_indices"], dtype=np.int64)
                        target = score_matrix[rec_idx, core_idx]
                        if score_matrix.shape[0] > 1:
                            other = np.max(np.delete(score_matrix[:, core_idx], rec_idx, axis=0), axis=0)
                        else:
                            other = np.zeros_like(target)
                        margin_values = target - other
                        acc = quality[family]
                        acc.unary_margin_values.append(float(np.mean(margin_values)))
                        acc.competing_margin_values.append(float(np.mean(margin_values)))
                        acc.unary_entropy_values.append(_softmax_entropy(QUALITY_LOGIT_SCALE * score_matrix[:, core_idx]))
                        acc.background_margin_values.append(float(np.mean(target - record["risk"][core_idx])))
                        acc.risk_weighted_unary_values.append(float(np.mean(target * (1.0 - record["risk"][core_idx]))))

                if int(args.progress_every_sources) > 0 and source_processed % int(args.progress_every_sources) == 0:
                    print(
                        json.dumps(
                            {
                                "phase": PHASE_ID,
                                "source_processed": source_processed,
                                "selected_core_processed": selected_core_processed,
                                "query_rows": len(query_rows),
                                "elapsed_sec": time.time() - started,
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )

    quality_rows_pre = [quality[family].row(family, confirmed_core_count, False) for family, _, is_method in QUERY_FAMILIES if is_method]
    method_candidates = [
        row for row in quality_rows_pre
        if row["query_count"] >= confirmed_core_count * 0.90
    ]
    if method_candidates:
        selected_quality = max(
            method_candidates,
            key=lambda row: (
                float(row["unary_margin_mean"]),
                -float(row["unary_entropy_mean"]),
                float(row["competing_object_margin_mean"]),
                -abs(1.0 - float(row["object_query_vs_mask_average_cosine_mean"])),
            ),
        )
        selected_query_family = str(selected_quality["query_family"])
    else:
        selected_query_family = "Q3_trimmed_core_mean"

    quality_rows = [quality[family].row(family, confirmed_core_count, family == selected_query_family) for family, _, _ in QUERY_FAMILIES]
    selected_row = next(row for row in quality_rows if row["query_family"] == selected_query_family)
    for row in query_rows:
        row["selected_for_expansion"] = row["query_family"] == selected_query_family

    vectors_arr = np.stack(vectors, axis=0).astype(np.float32) if vectors else np.zeros((0, feature_dim), dtype=np.float32)
    vector_npz = out / "object_query_vectors.npz"
    np.savez_compressed(
        vector_npz,
        query_vectors=vectors_arr,
        query_ids=np.asarray(query_ids, dtype="U512"),
        feature_names=np.asarray(FEATURE_NAMES, dtype="U64"),
        schema_version=np.asarray("stream4d_v95_phase3_query_vectors_v1"),
        uses_gt_for_prediction=np.asarray(False),
        uses_future=np.asarray(False),
    )
    vector_sha = _sha256(vector_npz)
    for row in vector_index_rows:
        row["sha256"] = vector_sha

    unary_fields = [
        "schema_version",
        "phase_id",
        "run_id",
        "query_family",
        "query_vector_ref",
        "core_variant_id",
        "scene_id",
        "window_id",
        "frame_id",
        "source_mask_id",
        "object_id",
        "region_id",
        "region_index",
        "proto_similarity",
        "D4RT_witness_score",
        "D4RT_negative_score",
        "edge_inside_score",
        "risk_score",
        "background_score",
        "object_unary_score",
        "rank_within_source",
        "source_object_count",
        "is_core_region",
        "uses_gt_for_prediction",
        "uses_future",
    ]
    unary_path = out / "region_object_unary_rows.csv"
    unary_path.parent.mkdir(parents=True, exist_ok=True)
    unary_rows = 0
    with unary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=unary_fields)
        writer.writeheader()
        for shard_path in shard_paths:
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
                    key = _source_key_parts(raw_source_key)
                    source_cores = selected_by_source.get(key, [])
                    if not source_cores:
                        continue
                    meta = phase1_sources.get(key)
                    node_map = nodes_by_source.get(key, {})
                    if not meta or not node_map:
                        continue
                    region_mask = region_source_index == source_idx
                    source_region_indices = region_indices_all[region_mask]
                    source_region_ids = [region_ids_all[pos] for pos in np.nonzero(region_mask)[0].tolist()]
                    nodes = [node_map.get(int(region_index)) for region_index in source_region_indices.tolist()]
                    if any(node is None for node in nodes):
                        continue
                    object_positions = np.nonzero(object_source_index == source_idx)[0]
                    if object_positions.size == 0:
                        continue
                    object_pairs = sorted((int(object_local_index[pos]), object_keys[pos]) for pos in object_positions.tolist())
                    object_id_to_local = {object_id: local_idx for local_idx, object_id in object_pairs}
                    k_objects = max(object_id_to_local.values()) + 1
                    r_regions = len(source_region_indices)
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
                    feat = _node_features([node for node in nodes if node is not None], _num(meta.get("mask_area_px"), 1.0))
                    region_index_to_local = {int(region_index): idx for idx, region_index in enumerate(source_region_indices.tolist())}
                    source_object_count = sum(1 for core in source_cores if str(core["object_id"]) in object_id_to_local)

                    for core_row in source_cores:
                        object_id = str(core_row["object_id"])
                        if object_id not in object_id_to_local:
                            continue
                        local_core = [region_index_to_local[idx] for idx in core_row["query_core_region_indices"] if idx in region_index_to_local]
                        if not local_core:
                            continue
                        object_local_idx = object_id_to_local[object_id]
                        matrix, obj_score, margin, _other = _region_feature_matrix(
                            scores,
                            int(object_local_idx),
                            feat,
                            _int(core_row.get("temporal_support_proxy"), 0),
                        )
                        query_vectors = _query_vectors_for_core(matrix, local_core, obj_score, margin, feat)
                        query = query_vectors[selected_query_family]
                        mask_query = query_vectors["Q0_mask_average_prototype_control"]
                        score_matrix_input = matrix if selected_query_family == "Q0_mask_average_prototype_control" else matrix - mask_query
                        proto = _cosine_scores(score_matrix_input, query, device)
                        object_unary = (
                            0.70 * proto
                            + 0.16 * feat["d4rt"]
                            + 0.08 * np.clip(margin, 0.0, 1.0)
                            - 0.12 * feat["risk"]
                            - 0.06 * feat["negative"]
                        )
                        ranks = np.empty_like(object_unary, dtype=np.int32)
                        ranks[np.argsort(object_unary)[::-1]] = np.arange(1, len(object_unary) + 1, dtype=np.int32)
                        query_ref = query_ref_by_key[(str(core_row["variant_id"]), key[0], key[1], key[2], key[3], object_id, selected_query_family)]
                        core_set = set(local_core)
                        for local_idx in range(r_regions):
                            writer.writerow(
                                {
                                    "schema_version": "stream4d_v95_phase3_region_object_unary_v1",
                                    "phase_id": PHASE_ID,
                                    "run_id": RUN_ID,
                                    "query_family": selected_query_family,
                                    "query_vector_ref": query_ref,
                                    "core_variant_id": core_row["variant_id"],
                                    "scene_id": key[0],
                                    "window_id": key[1],
                                    "frame_id": key[2],
                                    "source_mask_id": key[3],
                                    "object_id": object_id,
                                    "region_id": source_region_ids[local_idx],
                                    "region_index": int(source_region_indices[local_idx]),
                                    "proto_similarity": float(proto[local_idx]),
                                    "D4RT_witness_score": float(feat["d4rt"][local_idx]),
                                    "D4RT_negative_score": float(feat["negative"][local_idx]),
                                    "edge_inside_score": float(1.0 - np.clip(feat["near_edge"][local_idx], 0.0, 1.0)),
                                    "risk_score": float(feat["risk"][local_idx]),
                                    "background_score": float(feat["risk"][local_idx] + feat["negative"][local_idx]),
                                    "object_unary_score": float(object_unary[local_idx]),
                                    "rank_within_source": int(ranks[local_idx]),
                                    "source_object_count": source_object_count,
                                    "is_core_region": local_idx in core_set,
                                    "uses_gt_for_prediction": False,
                                    "uses_future": False,
                                }
                            )
                            unary_rows += 1

    phase3_pass = bool(
        float(selected_row["query_count"]) >= confirmed_core_count * 0.90
        and float(selected_row["unary_margin_mean"]) >= 0.05
        and float(selected_row["unary_entropy_mean"]) <= 0.75
        and float(selected_row["competing_object_margin_mean"]) >= 0.03
    )
    blocker = ""
    if not phase3_pass:
        if float(selected_row["object_query_vs_mask_average_cosine_mean"]) >= 0.98:
            blocker = "OBJECT_QUERY_CORE_NEAR_MASK_AVERAGE"
        elif float(selected_row["unary_entropy_mean"]) > 0.75:
            blocker = "OBJECT_QUERY_ENTROPY_UNSTABLE"
        elif float(selected_row["competing_object_margin_mean"]) < 0.03:
            blocker = "OBJECT_QUERY_COMPETING_MARGIN_LOW"
        else:
            blocker = "OBJECT_QUERY_BLOCKER"
    if (
        phase3_pass
        and float(selected_row["object_query_vs_mask_average_cosine_mean"]) >= 0.97
        and float(selected_row["unary_margin_mean"]) >= 0.05
    ):
        blocker = "QUERY_MARGIN_MAY_BE_CONTROL_BIAS"

    gate_rows = [
        {
            "schema_version": "stream4d_v95_phase3_gate_v1",
            "phase_id": PHASE_ID,
            "run_id": RUN_ID,
            "criterion": "query_count_ge_selected_confirmed_core_count_x_0p90",
            "pass": float(selected_row["query_count"]) >= confirmed_core_count * 0.90,
            "observed": selected_row["query_count"],
            "required": confirmed_core_count * 0.90,
            "uses_gt_for_prediction": False,
            "uses_future": False,
        },
        {
            "schema_version": "stream4d_v95_phase3_gate_v1",
            "phase_id": PHASE_ID,
            "run_id": RUN_ID,
            "criterion": "unary_margin_mean_ge_0p05",
            "pass": float(selected_row["unary_margin_mean"]) >= 0.05,
            "observed": selected_row["unary_margin_mean"],
            "required": 0.05,
            "uses_gt_for_prediction": False,
            "uses_future": False,
        },
        {
            "schema_version": "stream4d_v95_phase3_gate_v1",
            "phase_id": PHASE_ID,
            "run_id": RUN_ID,
            "criterion": "unary_entropy_mean_le_0p75",
            "pass": float(selected_row["unary_entropy_mean"]) <= 0.75,
            "observed": selected_row["unary_entropy_mean"],
            "required": 0.75,
            "uses_gt_for_prediction": False,
            "uses_future": False,
        },
        {
            "schema_version": "stream4d_v95_phase3_gate_v1",
            "phase_id": PHASE_ID,
            "run_id": RUN_ID,
            "criterion": "competing_object_margin_mean_ge_0p03",
            "pass": float(selected_row["competing_object_margin_mean"]) >= 0.03,
            "observed": selected_row["competing_object_margin_mean"],
            "required": 0.03,
            "uses_gt_for_prediction": False,
            "uses_future": False,
        },
        {
            "schema_version": "stream4d_v95_phase3_gate_v1",
            "phase_id": PHASE_ID,
            "run_id": RUN_ID,
            "criterion": "uses_gt_for_prediction_count_eq_0",
            "pass": True,
            "observed": 0,
            "required": 0,
            "uses_gt_for_prediction": False,
            "uses_future": False,
        },
        {
            "schema_version": "stream4d_v95_phase3_gate_v1",
            "phase_id": PHASE_ID,
            "run_id": RUN_ID,
            "criterion": "uses_future_count_eq_0",
            "pass": True,
            "observed": 0,
            "required": 0,
            "uses_gt_for_prediction": False,
            "uses_future": False,
        },
    ]

    summary = {
        "schema": "stream4d_v95_phase3_object_query_summary_v1",
        "phase_id": PHASE_ID,
        "run_id": RUN_ID,
        "created_at": created_at,
        "decision": "PASS_V95_PHASE3_OBJECT_QUERY_READY" if phase3_pass else "NO_GO_V95_PHASE3_OBJECT_QUERY",
        "blocker": blocker,
        "core_root": _rel(core_root),
        "field_root": _rel(field_root),
        "raw_phase2_confirmed_core_count": core_meta["raw_phase2_confirmed_core_count"],
        "selected_confirmed_core_count": confirmed_core_count,
        "query_count": int(selected_row["query_count"]),
        "query_row_count": len(query_rows),
        "query_feature_dim": feature_dim,
        "selected_query_family": selected_query_family,
        "query_norm_mean": selected_row["query_norm_mean"],
        "query_entropy_mean": selected_row["query_entropy_mean"],
        "query_margin_proxy_mean": selected_row["query_margin_proxy_mean"],
        "query_coverage_rate": selected_row["query_coverage_rate"],
        "unary_margin_mean": selected_row["unary_margin_mean"],
        "unary_entropy_mean": selected_row["unary_entropy_mean"],
        "background_unary_margin": selected_row["background_unary_margin"],
        "competing_object_margin_mean": selected_row["competing_object_margin_mean"],
        "object_query_vs_mask_average_cosine_mean": selected_row["object_query_vs_mask_average_cosine_mean"],
        "risk_weighted_unary_mean": selected_row["risk_weighted_unary_mean"],
        "source_processed": source_processed,
        "selected_core_processed": selected_core_processed,
        "score_backend_counts": dict(score_backend_counts),
        "uses_gt_for_prediction_count": 0,
        "uses_future_count": 0,
        "uses_gt_for_prediction": False,
        "uses_future": False,
        "diagnostic_auc_available": False,
        "region_retrieval_AUC_diagnostic": "",
        "same_gt_region_topK_purity_diagnostic": "",
        "same_semantic_hard_negative_AUC_diagnostic": "",
        "query_to_GT_region_precision_at_10_diagnostic": "",
        "query_to_GT_region_precision_at_50_diagnostic": "",
        "duration_sec": time.time() - started,
        "row_counts": {
            "object_query_rows": len(query_rows),
            "object_query_vector_index_rows": len(vector_index_rows),
            "query_quality_rows": len(quality_rows),
            "query_casebook_rows": len(casebook_rows),
            "query_failure_rows": len(failure_rows),
            "region_object_unary_rows": unary_rows,
            "variant_gate_rows": len(gate_rows),
        },
    }

    _write_csv(out / "object_query_rows.csv", query_rows)
    _write_csv(out / "object_query_vector_index.csv", vector_index_rows)
    _write_csv(out / "query_quality_rows.csv", quality_rows)
    _write_csv(out / "query_casebook_rows.csv", casebook_rows)
    _write_csv(out / "query_failure_rows.csv", failure_rows)
    _write_csv(out / "variant_gate_rows.csv", gate_rows)
    _write_json(out / "summary.json", summary)
    outputs = [
        out / "object_query_rows.csv",
        out / "object_query_vector_index.csv",
        out / "object_query_vectors.npz",
        out / "region_object_unary_rows.csv",
        out / "query_quality_rows.csv",
        out / "query_casebook_rows.csv",
        out / "query_failure_rows.csv",
        out / "variant_gate_rows.csv",
        out / "summary.json",
    ]
    _write_json(out / "SHA256SUMS.json", {_rel(path): _sha256(path) for path in outputs if path.exists()})
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True), flush=True)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--core-root", default=str(DEFAULT_CORE_ROOT))
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
