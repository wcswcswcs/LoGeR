#!/usr/bin/env python3
"""Run v95 Phase4E GPU object-axis/core-query readout.

This repair branch avoids the large Phase3 region-object CSV during method
materialization. It reads the object-axis NPZ shards directly, computes
object/core-conditioned ownership scores with torch on the selected device, and
only writes evaluator-facing CSV artifacts after masks are materialized.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import build_v95_phase3_object_query as phase3  # noqa: E402
from tools import build_v95_phase4_core_conditioned_expansion as base  # noqa: E402
from tools import run_v90_dev_extent_score_cross_audit as phase7d  # noqa: E402
from tools import run_v91_phase4_radius_sweep as radius_sweep  # noqa: E402


PHASE_ID = "v95_phase4E_gpu_object_axis_readout"
RUN_ID = "v95_phase4E_gpu_object_axis_readout"
OUT = ROOT / "outputs/audit/v95_phase4E_gpu_object_axis_readout"
PHASE0 = ROOT / "outputs/audit/v95_phase0_fact_lock/summary.json"
PHASE1 = ROOT / "outputs/audit/v95_phase1_physical_source_registry"
PHASE2 = ROOT / "outputs/audit/v95_phase2_object_core_discovery_repair1"
FIELD_ROOT = ROOT / "outputs/audit/v94_phase7c_object_axis_field_full_dev_combined"


def _variant_specs() -> list[dict[str, Any]]:
    return [
        {
            "variant_id": "E0_axis_margin_precision",
            "family": "real",
            "score_mode": "axis",
            "margin_floor": 0.08,
            "risk_cap": 0.40,
            "score_floor": 0.10,
            "area_cap": 0.34,
            "description": "Strict object-axis margin with broad/high-risk suppression.",
        },
        {
            "variant_id": "E1_axis_balanced",
            "family": "real",
            "score_mode": "axis",
            "margin_floor": 0.04,
            "risk_cap": 0.52,
            "score_floor": 0.02,
            "area_cap": 0.46,
            "description": "Balanced object-axis readout without source-preserve fallback.",
        },
        {
            "variant_id": "E2_core_query_precision",
            "family": "real",
            "score_mode": "query",
            "margin_floor": 0.06,
            "risk_cap": 0.44,
            "score_floor": 0.08,
            "area_cap": 0.36,
            "description": "Source-centered core-query score with strict margin/risk gate.",
        },
        {
            "variant_id": "E3_core_axis_hybrid",
            "family": "real",
            "score_mode": "hybrid",
            "margin_floor": 0.03,
            "risk_cap": 0.55,
            "score_floor": 0.02,
            "area_cap": 0.50,
            "description": "Hybrid raw object-axis and core-query score.",
        },
        {
            "variant_id": "E4_ap50_precision",
            "family": "real",
            "score_mode": "hybrid",
            "margin_floor": 0.10,
            "risk_cap": 0.36,
            "score_floor": 0.12,
            "area_cap": 0.28,
            "fallback_fraction": 0.0,
            "description": "AP50-oriented compact readout; no broad/shared positive bridge.",
        },
        {
            "variant_id": "E5_query_source_preserve",
            "family": "real",
            "score_mode": "query",
            "margin_floor": 0.02,
            "risk_cap": 0.72,
            "score_floor": 0.00,
            "area_cap": 0.82,
            "fallback_fraction": 0.22,
            "description": "GPU query readout plus bounded source-local fallback.",
        },
        {
            "variant_id": "E6_hybrid_uncertain_preserve",
            "family": "real",
            "score_mode": "hybrid",
            "margin_floor": 0.02,
            "risk_cap": 0.75,
            "score_floor": 0.00,
            "area_cap": 0.90,
            "fallback_fraction": 0.35,
            "description": "GPU hybrid readout with C4-style uncertain source-local fallback.",
        },
    ]


def _device(name: str) -> torch.device:
    if name.startswith("cuda") and torch.cuda.is_available():
        return torch.device(name)
    return torch.device("cpu")


def _mask_content_sha(mask: np.ndarray) -> str:
    import hashlib

    h = hashlib.sha256()
    h.update(np.asarray(mask, dtype=np.uint8).tobytes(order="C"))
    return h.hexdigest()


def _query_consistency_by_object(query_root: Path, query_family: str) -> dict[str, float]:
    vector_path = query_root / "object_query_vectors.npz"
    index_path = query_root / "object_query_vector_index.csv"
    rows_path = query_root / "object_query_rows.csv"
    if not vector_path.exists() or not index_path.exists() or not rows_path.exists():
        return {}
    with np.load(vector_path, allow_pickle=False) as data:
        vectors = data["query_vectors"].astype(np.float32)
    ref_to_index: dict[str, int] = {}
    with index_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("query_family") == query_family:
                ref_to_index[str(row["query_vector_ref"])] = int(row["vector_index"])
    by_object: dict[str, list[int]] = defaultdict(list)
    with rows_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("selected_for_expansion") != "True":
                continue
            idx = ref_to_index.get(str(row.get("query_vector_ref", "")))
            if idx is None:
                continue
            by_object[str(row["object_id"])].append(idx)
    consistency: dict[str, float] = {}
    for object_id, indices in by_object.items():
        x = vectors[indices]
        x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-8)
        n = int(x.shape[0])
        if n <= 1:
            consistency[object_id] = 1.0
        else:
            consistency[object_id] = float((np.sum(x @ x.T) - n) / max(1, n * (n - 1)))
    return consistency


def _allowed_objects(
    selected_by_source: dict[tuple[str, str, int, int], list[dict[str, Any]]],
    min_support: int,
    query_consistency: dict[str, float],
    min_query_consistency: float,
) -> set[str] | None:
    if min_support <= 1 and min_query_consistency <= -1.0:
        return None
    counts: Counter[str] = Counter()
    for rows in selected_by_source.values():
        for row in rows:
            counts[str(row["object_id"])] += 1
    return {
        object_id
        for object_id, count in counts.items()
        if count >= min_support and query_consistency.get(object_id, 1.0) >= min_query_consistency
    }


def _load_phase1_sources(path: Path) -> dict[tuple[str, str, int, int], dict[str, str]]:
    sources: dict[tuple[str, str, int, int], dict[str, str]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if base._bool(row.get("uses_gt_for_prediction")) or base._bool(row.get("uses_future")):
                continue
            sources[(row["scene_id"], row["window_id"], base._int(row["frame_id"]), base._int(row["source_mask_id"]))] = dict(row)
    return sources


def _stack_unary(
    k_objects: int,
    r_regions: int,
    unary_object_local_index: np.ndarray,
    unary_region_local_index: np.ndarray,
    unary_cosine: np.ndarray,
    source_unary_mask: np.ndarray,
    device: torch.device,
) -> torch.Tensor:
    scores = torch.full((int(k_objects), int(r_regions)), -1.0, dtype=torch.float32, device=device)
    obj_idx = torch.as_tensor(unary_object_local_index[source_unary_mask], dtype=torch.long, device=device)
    reg_idx = torch.as_tensor(unary_region_local_index[source_unary_mask], dtype=torch.long, device=device)
    values = torch.as_tensor(unary_cosine[source_unary_mask], dtype=torch.float32, device=device)
    keep = (obj_idx >= 0) & (obj_idx < k_objects) & (reg_idx >= 0) & (reg_idx < r_regions)
    if bool(torch.any(keep)):
        scores[obj_idx[keep], reg_idx[keep]] = values[keep]
    return scores


def _source_features(
    nodes: list[dict[str, str]],
    source_area: float,
    device: torch.device,
) -> tuple[dict[str, np.ndarray], dict[str, torch.Tensor]]:
    feat_np = phase3._node_features(nodes, source_area)
    feat_t = {name: torch.as_tensor(value, dtype=torch.float32, device=device) for name, value in feat_np.items()}
    return feat_np, feat_t


def _query_proto_scores(
    matrix: np.ndarray,
    core_local: list[int],
    obj_score: np.ndarray,
    margin: np.ndarray,
    feat_np: dict[str, np.ndarray],
    selected_query: str,
    device: torch.device,
) -> np.ndarray:
    vectors = phase3._query_vectors_for_core(matrix, core_local, obj_score, margin, feat_np)
    query = vectors[selected_query]
    mask_query = vectors["Q0_mask_average_prototype_control"]
    x_np = matrix - mask_query
    x = torch.as_tensor(x_np, dtype=torch.float32, device=device)
    q = torch.as_tensor(query, dtype=torch.float32, device=device)
    x = torch.nn.functional.normalize(x, p=2, dim=1, eps=1e-8)
    q = torch.nn.functional.normalize(q, p=2, dim=0, eps=1e-8)
    return torch.mv(x, q).detach().cpu().numpy().astype(np.float32)


def _candidate_score(
    raw: np.ndarray,
    margin: np.ndarray,
    proto: np.ndarray,
    feat_np: dict[str, np.ndarray],
    spec: dict[str, Any],
) -> np.ndarray:
    mode = str(spec["score_mode"])
    if mode == "axis":
        score = 0.64 * raw + 0.22 * np.clip(margin, 0.0, 1.0) + 0.10 * feat_np["d4rt"] - 0.18 * feat_np["risk"] - 0.08 * feat_np["negative"]
    elif mode == "query":
        score = 0.70 * proto + 0.12 * np.clip(margin, 0.0, 1.0) + 0.12 * feat_np["d4rt"] - 0.18 * feat_np["risk"] - 0.08 * feat_np["negative"]
    else:
        score = 0.38 * raw + 0.38 * proto + 0.14 * np.clip(margin, 0.0, 1.0) + 0.10 * feat_np["d4rt"] - 0.20 * feat_np["risk"] - 0.08 * feat_np["negative"]
    return score.astype(np.float32)


def _select_regions(
    candidate_rows: list[dict[str, Any]],
    feat_np: dict[str, np.ndarray],
    spec: dict[str, Any],
) -> dict[str, set[int]]:
    selected: dict[str, set[int]] = {row["object_id"]: set(row["core_region_indices"]) for row in candidate_rows}
    if not candidate_rows:
        return selected
    scores = np.stack([row["score"] for row in candidate_rows], axis=0)
    margins = np.stack([row["margin"] for row in candidate_rows], axis=0)
    best_idx = np.argmax(scores, axis=0)
    second = np.partition(scores, -2, axis=0)[-2] if scores.shape[0] > 1 else np.full(scores.shape[1], -999.0, dtype=np.float32)
    best_score = scores[best_idx, np.arange(scores.shape[1])]
    best_margin = margins[best_idx, np.arange(scores.shape[1])]
    risk = feat_np["risk"]
    accept = (
        (best_score >= float(spec["score_floor"]))
        & (best_margin >= float(spec["margin_floor"]))
        & ((best_score - second) >= max(0.0, float(spec["margin_floor"]) * 0.5))
        & (risk <= float(spec["risk_cap"]))
    )
    for local_idx, ok in enumerate(accept.tolist()):
        if not ok:
            continue
        object_id = candidate_rows[int(best_idx[local_idx])]["object_id"]
        selected[object_id].add(int(candidate_rows[int(best_idx[local_idx])]["region_indices"][local_idx]))
    fallback_fraction = float(spec.get("fallback_fraction", 0.0))
    if fallback_fraction > 0.0:
        for row in candidate_rows:
            object_id = row["object_id"]
            target_n = max(len(selected[object_id]), int(math.ceil(fallback_fraction * len(row["region_indices"]))))
            if target_n <= len(selected[object_id]):
                continue
            order = np.argsort(row["score"])[::-1]
            for local_idx in order.tolist():
                if len(selected[object_id]) >= target_n:
                    break
                if feat_np["risk"][local_idx] > float(spec["risk_cap"]):
                    continue
                selected[object_id].add(int(row["region_indices"][local_idx]))
    return selected


def _object_score(selected_rows: list[dict[str, Any]]) -> float:
    return float(
        np.clip(
            0.50
            + 0.32 * base._mean([base._num(row.get("mean_score")) for row in selected_rows])
            + 0.16 * base._mean([base._num(row.get("mean_margin")) for row in selected_rows])
            - 0.14 * base._mean([base._num(row.get("mean_risk")) for row in selected_rows]),
            0.0,
            1.0,
        )
    )


def _gate_rows(
    variant_metric_rows: list[dict[str, Any]],
    specs: list[dict[str, Any]],
    phase0: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], bool, dict[str, Any]]:
    spec_by_id = {spec["variant_id"]: spec for spec in specs}
    required_ap = base._num(phase0.get("required_MV_AP_window"))
    required_ap50 = base._num(phase0.get("required_MV_AP50_window"))
    candidate_required_ap = max(base._num(phase0.get("v91_best_MV_AP_window")) + 0.002, base._num(phase0.get("best_control_MV_AP_window")) + 0.005)
    candidate_required_ap50 = max(base._num(phase0.get("v91_best_MV_AP50_window")) + 0.004, base._num(phase0.get("best_control_MV_AP50_window")) + 0.010)
    gate_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    best_real: dict[str, Any] = {}
    any_pass = False
    for row in variant_metric_rows:
        variant_id = str(row.get("variant_id", ""))
        spec = spec_by_id.get(variant_id, {})
        mv_ap = base._num(row.get("mean_MV_AP_window"))
        mv_ap50 = base._num(row.get("mean_MV_AP50_window"))
        collision = base._int(row.get("same_frame_collision_count"))
        missing = base._int(row.get("missing_mask_raster_count"))
        progress_gate = bool(mv_ap >= candidate_required_ap and mv_ap50 >= candidate_required_ap50)
        final_gate = bool(mv_ap >= required_ap and mv_ap50 >= required_ap50)
        provenance_gate = bool(collision == 0 and missing == 0 and not base._bool(row.get("uses_gt_for_prediction")) and not base._bool(row.get("uses_future")))
        pass_gate = bool(spec.get("family") == "real" and progress_gate and provenance_gate)
        any_pass = any_pass or pass_gate
        if spec.get("family") == "real" and (
            not best_real
            or (mv_ap, mv_ap50) > (base._num(best_real.get("mean_MV_AP_window"), -999.0), base._num(best_real.get("mean_MV_AP50_window"), -999.0))
        ):
            best_real = dict(row)
        gate_rows.append(
            {
                "schema_version": "stream4d_v95_phase4E_variant_gate_v1",
                "phase_id": PHASE_ID,
                "run_id": RUN_ID,
                "variant_id": variant_id,
                "family": spec.get("family", ""),
                "MV_AP_window": mv_ap,
                "MV_AP50_window": mv_ap50,
                "MV_AP25_window": row.get("mean_MV_AP25_window", ""),
                "ScoreFreeMatch50_window": row.get("mean_score_free_Match50_window", ""),
                "candidate_required_MV_AP_window": candidate_required_ap,
                "candidate_required_MV_AP50_window": candidate_required_ap50,
                "final_required_MV_AP_window": required_ap,
                "final_required_MV_AP50_window": required_ap50,
                "phase4_candidate_gate_pass": pass_gate,
                "phase4_final_threshold_gate_pass": final_gate,
                "progress_gate_pass": progress_gate,
                "provenance_gate_pass": provenance_gate,
                "same_frame_collision_count": collision,
                "missing_mask_raster_count": missing,
                "uses_gt_for_prediction": False,
                "uses_future": False,
            }
        )
        if spec.get("family") == "real" and not pass_gate:
            failure_rows.append(
                {
                    "schema_version": "stream4d_v95_phase4E_failure_v1",
                    "phase_id": PHASE_ID,
                    "run_id": RUN_ID,
                    "variant_id": variant_id,
                    "failure_type": "PHASE4_FAMILY_E_NO_CANDIDATE_GATE",
                    "MV_AP_window": mv_ap,
                    "MV_AP50_window": mv_ap50,
                    "repair_direction": (
                        "GPU direct object-axis readout did not reach candidate gate; "
                        "return to object-core specificity or object-axis candidate generation."
                    ),
                    "uses_gt_for_prediction": False,
                    "uses_future": False,
                }
            )
    return gate_rows, failure_rows, any_pass, best_real


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    out = base._resolve(args.output_root)
    if out.exists() and args.clean:
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    device = _device(str(args.device))
    phase0 = json.loads(PHASE0.read_text(encoding="utf-8"))
    core_root = base._resolve(args.core_root)
    field_root = base._resolve(args.field_root)
    phase2_summary = json.loads((core_root / "summary.json").read_text(encoding="utf-8"))
    if phase2_summary.get("decision") != "PASS_V95_PHASE2_OBJECT_CORE_READY":
        raise RuntimeError("v95 Phase2 must pass before Phase4E")
    selected_by_source, _selected_regions, core_meta = phase3._load_selected_cores(core_root, int(args.max_sources))
    query_consistency = _query_consistency_by_object(base._resolve(args.query_root), str(args.selected_query_family))
    allowed = _allowed_objects(
        selected_by_source,
        int(args.min_query_support),
        query_consistency,
        float(args.min_object_query_consistency),
    )
    selected_keys = set(selected_by_source)
    source_meta = _load_phase1_sources(PHASE1 / "source_container_rows.csv")
    nodes_by_source = phase3._load_region_nodes(PHASE1 / "region_node_rows.csv", selected_keys)
    shard_paths = sorted((field_root / "field_shards").glob("object_axis_unary_shard_*.npz"))
    if int(args.max_shards) > 0:
        shard_paths = shard_paths[: int(args.max_shards)]
    if not shard_paths:
        raise RuntimeError(f"no object-axis shards under {field_root / 'field_shards'}")

    specs = _variant_specs()
    variant_ids = [spec["variant_id"] for spec in specs]
    frame_writer = base.FrameWriter(out, variant_ids)
    label_cache: dict[tuple[str, int], np.ndarray] = {}
    generated_rows: list[dict[str, Any]] = []
    mv_rows: list[dict[str, Any]] = []
    ownership_rows: list[dict[str, Any]] = []
    gpu_audit_rows: list[dict[str, Any]] = []
    area_ratios_by_variant: dict[str, list[float]] = defaultdict(list)
    core_retention_by_variant: dict[str, list[float]] = defaultdict(list)
    selected_counts_by_variant: dict[str, list[int]] = defaultdict(list)
    source_processed = 0
    selected_core_processed = 0
    score_backend_counts: Counter[str] = Counter()
    created_at = base._created_at()
    config_rows = [
        {
            "schema_version": "stream4d_v95_phase4E_variant_config_v1",
            "phase_id": PHASE_ID,
            "run_id": RUN_ID,
            "created_at": created_at,
            **spec,
            "core_root": base._rel(core_root),
            "field_root": base._rel(field_root),
            "min_query_support": int(args.min_query_support),
            "selected_query_family": str(args.selected_query_family),
            "query_root": base._rel(base._resolve(args.query_root)),
            "min_object_query_consistency": float(args.min_object_query_consistency),
            "device_requested": str(args.device),
            "device_resolved": str(device),
            "uses_gt_for_prediction": False,
            "uses_future": False,
        }
        for spec in specs
    ]

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
                key = phase3._source_key_parts(raw_source_key)
                source_cores = selected_by_source.get(key, [])
                if not source_cores:
                    continue
                meta = source_meta.get(key)
                node_map = nodes_by_source.get(key, {})
                if not meta or not node_map:
                    continue
                source_cores = [
                    row for row in source_cores
                    if allowed is None or str(row["object_id"]) in allowed
                ]
                if not source_cores:
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
                usable_cores = [row for row in source_cores if str(row["object_id"]) in object_id_to_local]
                if not usable_cores:
                    continue
                scene, window, frame_id, mask_id = key
                mask_path = base._resolve(meta.get("mask_path", ""))
                frame_key = (scene, int(frame_id))
                if frame_key not in label_cache:
                    label_cache[frame_key] = base._read_label(mask_path)
                source_mask = label_cache[frame_key] == int(mask_id)
                if not np.any(source_mask):
                    continue
                frame_writer.ensure_frame(scene, int(frame_id), source_mask.shape)
                source_area = float(max(1, int(np.count_nonzero(source_mask))))
                feat_np, _feat_t = _source_features([node for node in nodes if node is not None], source_area, device)
                k_objects = max(object_id_to_local.values()) + 1
                r_regions = len(source_region_indices)
                source_unary = unary_source_index == source_idx
                scores_t = _stack_unary(
                    k_objects,
                    r_regions,
                    unary_object_local_index,
                    unary_region_local_index,
                    unary_cosine,
                    source_unary,
                    device,
                )
                if not bool(torch.any(scores_t >= 0.0)):
                    continue
                scores_np = scores_t.detach().cpu().numpy().astype(np.float32)
                region_index_to_local = {int(region_index): idx for idx, region_index in enumerate(source_region_indices.tolist())}
                all_candidate_rows: list[dict[str, Any]] = []
                for core_row in usable_cores:
                    object_id = str(core_row["object_id"])
                    object_local_idx = int(object_id_to_local[object_id])
                    local_core = [region_index_to_local[idx] for idx in core_row["query_core_region_indices"] if idx in region_index_to_local]
                    if not local_core:
                        continue
                    matrix, obj_score, margin, _other = phase3._region_feature_matrix(
                        scores_np,
                        object_local_idx,
                        feat_np,
                        phase3._int(core_row.get("temporal_support_proxy"), 0),
                    )
                    proto = _query_proto_scores(matrix, local_core, obj_score, margin, feat_np, str(args.selected_query_family), device)
                    raw = np.clip(scores_np[object_local_idx], 0.0, 1.0)
                    if scores_np.shape[0] > 1:
                        other = np.max(np.concatenate([scores_np[:object_local_idx], scores_np[object_local_idx + 1:]], axis=0), axis=0)
                    else:
                        other = np.zeros_like(raw)
                    axis_margin = raw - np.clip(other, 0.0, 1.0)
                    all_candidate_rows.append(
                        {
                            "object_id": object_id,
                            "region_indices": source_region_indices.tolist(),
                            "core_region_indices": [int(source_region_indices[i]) for i in local_core],
                            "raw": raw,
                            "margin": axis_margin.astype(np.float32),
                            "proto": proto,
                            "query_margin_mean": phase3._mean([float(margin[i]) for i in local_core]),
                        }
                    )
                    selected_core_processed += 1

                if not all_candidate_rows:
                    continue
                source_processed += 1
                score_backend_counts[str(device)] += 1
                for spec in specs:
                    spec_candidates: list[dict[str, Any]] = []
                    for row in all_candidate_rows:
                        score = _candidate_score(row["raw"], row["margin"], row["proto"], feat_np, spec)
                        spec_candidates.append({**row, "score": score})
                    selected_by_object = _select_regions(spec_candidates, feat_np, spec)
                    stat_by_object = {
                        row["object_id"]: {
                            "score": row["score"],
                            "margin": row["margin"],
                        }
                        for row in spec_candidates
                    }
                    for object_id, selected in selected_by_object.items():
                        if not selected:
                            continue
                        candidate = next(row for row in spec_candidates if row["object_id"] == object_id)
                        selected = base._cap_by_area(
                            [
                                {"region_index": int(region_idx), "region_area_ratio": float(feat_np["area_ratio"][local_idx])}
                                for local_idx, region_idx in enumerate(candidate["region_indices"])
                            ],
                            selected,
                            float(spec["area_cap"]),
                        )
                        if not selected:
                            continue
                        mask = base._node_mask(node_map, selected, source_mask)
                        new_id = frame_writer.add_mask(str(spec["variant_id"]), mask)
                        if new_id <= 0:
                            continue
                        selected_area = int(np.count_nonzero(mask))
                        if selected_area <= 0:
                            continue
                        selected_local = [region_index_to_local[idx] for idx in selected if idx in region_index_to_local]
                        selected_stats = stat_by_object[object_id]
                        mean_score = phase3._mean([float(selected_stats["score"][i]) for i in selected_local])
                        mean_margin = phase3._mean([float(selected_stats["margin"][i]) for i in selected_local])
                        mean_risk = phase3._mean([float(feat_np["risk"][i]) for i in selected_local])
                        core_set = set(candidate["core_region_indices"])
                        core_retention = len(core_set & selected) / max(1, len(core_set))
                        area_ratio = selected_area / source_area
                        area_ratios_by_variant[str(spec["variant_id"])].append(float(area_ratio))
                        core_retention_by_variant[str(spec["variant_id"])].append(float(core_retention))
                        selected_counts_by_variant[str(spec["variant_id"])].append(int(len(selected)))
                        object_score = _object_score([
                            {"mean_score": mean_score, "mean_margin": mean_margin, "mean_risk": mean_risk}
                        ])
                        gen_path = out / "generated_masks" / str(spec["variant_id"]) / scene / "mask" / f"{int(frame_id)}.png"
                        generated_rows.append(
                            {
                                "schema_version": "stream4d_v95_phase4E_generated_mask_v1",
                                "phase_id": PHASE_ID,
                                "run_id": RUN_ID,
                                "variant_id": spec["variant_id"],
                                "scene_id": scene,
                                "window_id": window,
                                "frame_id": int(frame_id),
                                "source_mask_id": int(mask_id),
                                "object_id": object_id,
                                "new_mask_id": int(new_id),
                                "generated_mask_path": base._rel(gen_path),
                                "source_mask_area": int(source_area),
                                "generated_mask_area": int(selected_area),
                                "generated_area_ratio": float(area_ratio),
                                "selected_region_count": int(len(selected)),
                                "core_region_count": int(len(core_set)),
                                "core_retention_rate": float(core_retention),
                                "mean_score": float(mean_score),
                                "mean_margin": float(mean_margin),
                                "mean_risk": float(mean_risk),
                                "mask_binary_sha256": _mask_content_sha(mask),
                                "uses_gt_for_prediction": False,
                                "uses_future": False,
                            }
                        )
                        mv_object_id = f"{spec['variant_id']}:{object_id}"
                        mv_rows.append(
                            {
                                "split": "dev",
                                "scene_id": scene,
                                "source_variant": spec["variant_id"],
                                "variant": spec["variant_id"],
                                "mv_object_id": mv_object_id,
                                "frame_id": int(frame_id),
                                "mask_id": int(new_id),
                                "frame_mask_score": object_score,
                                "object_score": object_score,
                                "uses_gt_for_prediction": False,
                                "uses_future": False,
                                "uses_rgbd_pose_mesh": False,
                                "materializable": True,
                                "selection_reason": f"v95_phase4E_gpu_{spec['score_mode']}_from_object_axis_npz",
                            }
                        )
                        ownership_rows.append(
                            {
                                "schema_version": "stream4d_v95_phase4E_ownership_audit_v1",
                                "phase_id": PHASE_ID,
                                "run_id": RUN_ID,
                                "variant_id": spec["variant_id"],
                                "scene_id": scene,
                                "window_id": window,
                                "frame_id": int(frame_id),
                                "source_mask_id": int(mask_id),
                                "object_id": object_id,
                                "selected_region_count": int(len(selected)),
                                "generated_area_ratio": float(area_ratio),
                                "core_retention_rate": float(core_retention),
                                "mean_score": float(mean_score),
                                "mean_margin": float(mean_margin),
                                "mean_risk": float(mean_risk),
                                "uses_gt_for_prediction": False,
                                "uses_future": False,
                            }
                        )
                if int(args.progress_every_sources) > 0 and source_processed % int(args.progress_every_sources) == 0:
                    print(
                        json.dumps(
                            {
                                "phase": PHASE_ID,
                                "source_processed": source_processed,
                                "selected_core_processed": selected_core_processed,
                                "generated_mask_rows": len(generated_rows),
                                "device": str(device),
                                "elapsed_sec": time.time() - started,
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    frame_writer.flush()
    gpu_audit_rows.append(
        {
            "schema_version": "stream4d_v95_phase4E_gpu_audit_v1",
            "phase_id": PHASE_ID,
            "run_id": RUN_ID,
            "device_requested": str(args.device),
            "device_resolved": str(device),
            "torch_cuda_available": torch.cuda.is_available(),
            "torch_cuda_device_count_visible": torch.cuda.device_count(),
            "score_backend_counts": json.dumps(dict(score_backend_counts), sort_keys=True),
            "uses_gt_for_prediction": False,
            "uses_future": False,
        }
    )
    base._write_csv(out / "variant_config_rows.csv", config_rows)
    base._write_csv(out / "gpu_audit_rows.csv", gpu_audit_rows)
    base._write_csv(out / "generated_mask_rows.csv", generated_rows)
    base._write_csv(out / "mv_object_rows.csv", [{"variant_id": row["variant"], "mv_object_id": row["mv_object_id"], "uses_gt_for_prediction": False, "uses_future": False} for row in mv_rows])
    base._write_csv(out / "mv_object_frame_mask_rows.csv", mv_rows)
    base._write_csv(out / "ownership_audit_rows.csv", ownership_rows)
    metric_rows: list[dict[str, Any]] = []
    case_rows: list[dict[str, Any]] = []
    if not bool(args.skip_eval):
        radius_sweep.OUT = out
        for spec in specs:
            variant_id = str(spec["variant_id"])
            rows = [row for row in mv_rows if row.get("variant") == variant_id]
            metrics, cases = radius_sweep._evaluate_variant(variant_id, rows)
            metric_rows.extend(metrics)
            case_rows.extend({**case, "variant_id": variant_id} for case in cases)
    aggregate_rows = phase7d._aggregate(metric_rows) if metric_rows else []
    aggregate_by_variant = {row.get("variant_id", ""): row for row in aggregate_rows}
    variant_metric_rows: list[dict[str, Any]] = []
    for spec in specs:
        variant_id = str(spec["variant_id"])
        agg = aggregate_by_variant.get(variant_id, {})
        variant_metric_rows.append(
            {
                "schema_version": "stream4d_v95_phase4E_variant_metric_v1",
                "phase_id": PHASE_ID,
                "run_id": RUN_ID,
                "variant_id": variant_id,
                "family": spec["family"],
                "mean_MV_AP_window": agg.get("mean_MV_AP_window", ""),
                "mean_MV_AP50_window": agg.get("mean_MV_AP50_window", ""),
                "mean_MV_AP25_window": agg.get("mean_MV_AP25_window", ""),
                "mean_score_free_Match50_window": agg.get("mean_score_free_Match50_window", ""),
                "mean_generated_area_ratio": base._mean(area_ratios_by_variant.get(variant_id, [])),
                "generated_area_ratio_p10": base._percentile(area_ratios_by_variant.get(variant_id, []), 10),
                "generated_area_ratio_p90": base._percentile(area_ratios_by_variant.get(variant_id, []), 90),
                "core_retention_rate": base._mean(core_retention_by_variant.get(variant_id, [])),
                "selected_region_count_mean": base._mean(selected_counts_by_variant.get(variant_id, [])),
                "same_frame_collision_count": agg.get("same_frame_collision_count", 0),
                "missing_mask_raster_count": agg.get("missing_mask_raster_count", 0),
                "uses_gt_for_prediction": False,
                "uses_future": False,
            }
        )
    gate_rows, failure_rows, any_pass, best_real = _gate_rows(variant_metric_rows, specs, phase0) if metric_rows else ([], [], False, {})
    base._write_csv(out / "mv_metric_rows.csv", metric_rows)
    base._write_csv(out / "mv_metric_aggregate_rows.csv", aggregate_rows)
    base._write_csv(out / "variant_metric_rows.csv", variant_metric_rows)
    base._write_csv(out / "variant_gate_rows.csv", gate_rows)
    base._write_csv(out / "variant_failure_rows.csv", failure_rows)
    base._write_csv(out / "casebook_rows.csv", case_rows)
    base._write_csv(out / "mv_iou_matrix_rows.csv", [])
    base._write_csv(out / "scorefree_match_rows.csv", [])
    summary = {
        "schema": "stream4d_v95_phase4E_gpu_object_axis_readout_summary_v1",
        "phase_id": PHASE_ID,
        "run_id": RUN_ID,
        "created_at": created_at,
        "decision": "PASS_V95_PHASE4_FAMILY_E_CANDIDATE_GATE" if any_pass else ("SMOKE_V95_PHASE4E_GPU_MATERIALIZATION_ONLY" if bool(args.skip_eval) else "NO_GO_V95_PHASE4_FAMILY_E_NO_CANDIDATE_GATE"),
        "family": "E_gpu_object_axis_core_query_readout",
        "core_root": base._rel(core_root),
        "field_root": base._rel(field_root),
        "min_query_support": int(args.min_query_support),
        "allowed_object_count": "" if allowed is None else len(allowed),
        "selected_query_family": str(args.selected_query_family),
        "query_root": base._rel(base._resolve(args.query_root)),
        "min_object_query_consistency": float(args.min_object_query_consistency),
        "query_consistency_available": bool(query_consistency),
        "source_processed": source_processed,
        "selected_core_processed": selected_core_processed,
        "device_requested": str(args.device),
        "device_resolved": str(device),
        "score_backend_counts": dict(score_backend_counts),
        "best_real_variant_id": best_real.get("variant_id", ""),
        "best_real_MV_AP_window": best_real.get("mean_MV_AP_window", ""),
        "best_real_MV_AP50_window": best_real.get("mean_MV_AP50_window", ""),
        "best_real_MV_AP25_window": best_real.get("mean_MV_AP25_window", ""),
        "best_real_ScoreFreeMatch50_window": best_real.get("mean_score_free_Match50_window", ""),
        "phase4_candidate_gate_pass": any_pass,
        "uses_gt_for_prediction_count": 0,
        "uses_future_count": 0,
        "uses_gt_for_prediction": False,
        "uses_future": False,
        "duration_sec": time.time() - started,
        "row_counts": {
            "variant_config_rows": len(config_rows),
            "gpu_audit_rows": len(gpu_audit_rows),
            "generated_mask_rows": len(generated_rows),
            "mv_object_rows": len(mv_rows),
            "mv_object_frame_mask_rows": len(mv_rows),
            "ownership_audit_rows": len(ownership_rows),
            "mv_metric_rows": len(metric_rows),
            "mv_metric_aggregate_rows": len(aggregate_rows),
            "variant_metric_rows": len(variant_metric_rows),
            "variant_gate_rows": len(gate_rows),
            "variant_failure_rows": len(failure_rows),
            "casebook_rows": len(case_rows),
        },
    }
    base._write_json(out / "summary.json", summary)
    outputs = [
        out / "variant_config_rows.csv",
        out / "gpu_audit_rows.csv",
        out / "generated_mask_rows.csv",
        out / "mv_object_rows.csv",
        out / "mv_object_frame_mask_rows.csv",
        out / "ownership_audit_rows.csv",
        out / "mv_metric_rows.csv",
        out / "mv_metric_aggregate_rows.csv",
        out / "variant_metric_rows.csv",
        out / "variant_gate_rows.csv",
        out / "variant_failure_rows.csv",
        out / "casebook_rows.csv",
        out / "summary.json",
    ]
    base._write_json(out / "SHA256SUMS.json", {base._rel(path): base._sha256(path) for path in outputs if path.exists()})
    print(json.dumps(base._jsonable(summary), indent=2, sort_keys=True), flush=True)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--core-root", default=str(PHASE2))
    parser.add_argument("--field-root", default=str(FIELD_ROOT))
    parser.add_argument("--query-root", default=str(ROOT / "outputs/audit/v95_phase3_object_query"))
    parser.add_argument("--output-root", default=str(OUT))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--selected-query-family", default="Q4_D4RT_weighted_core_mean")
    parser.add_argument("--min-query-support", type=int, default=30)
    parser.add_argument("--min-object-query-consistency", type=float, default=-1.0)
    parser.add_argument("--max-sources", type=int, default=0)
    parser.add_argument("--max-shards", type=int, default=0)
    parser.add_argument("--progress-every-sources", type=int, default=512)
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--clean", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
