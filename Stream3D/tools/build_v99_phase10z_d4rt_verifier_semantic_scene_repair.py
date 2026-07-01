#!/usr/bin/env python3
"""Use D4RT anchors as a verifier for Phase10P semantic scene stitching.

Phase10Y showed that D4RT-only local2history improves over controls slightly
but remains far below the scene gate. This phase reuses the already audited
chunk32/overlap3 self-stitched D4RT candidate table and tests two repair roles:

1. high-threshold D4RT-only adjacent stitching;
2. D4RT confirmation or supplement for Phase10P semantic local2history.

No DA3-D4RT mixed geometry edge is used here.
"""

from __future__ import annotations

import csv
import json
import math
import multiprocessing as mp
import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


STREAM3D_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = STREAM3D_ROOT.parent
if str(STREAM3D_ROOT) not in sys.path:
    sys.path.insert(0, str(STREAM3D_ROOT))

from tools import build_v99_phase1_f2_base_reproduction as p1  # noqa: E402
from tools import build_v99_phase10k_holdout_chunk_object_birth_sweep as p10k  # noqa: E402
from tools import build_v99_phase10o_overlap3_scene_stitch_repair as p10o  # noqa: E402


AUDIT_ROOT = STREAM3D_ROOT / "outputs/audit"
OUT_DIR = AUDIT_ROOT / "v99_phase10z_d4rt_verifier_semantic_scene_repair"
PHASE0_DIR = AUDIT_ROOT / "v99_phase0_fact_lock"
PHASE10O_DIR = AUDIT_ROOT / "v99_phase10o_overlap3_scene_stitch_repair"
PHASE10P_DIR = AUDIT_ROOT / "v99_phase10p_overlap3_scene_stitch_semantic_sweep"
PHASE10Y_DIR = AUDIT_ROOT / "v99_phase10y_d4rt_anchor_holdout_scene_stitch"
BASE_VARIANT = "O0_overlap3_chunk_birth_primary_emit"
P10P_BEST_VARIANT = "P_sem_tau0p70_gap99"
CHUNK_SIZE = 32
OVERLAP = 3
DEFAULT_EVAL_WORKERS = 8


def _rel(path: Path | str) -> str:
    q = Path(path)
    try:
        return q.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return q.as_posix()


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    if isinstance(value, Path):
        return _rel(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(data), indent=2, sort_keys=True) + "\n", encoding="utf-8")


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
            writer.writerow({field: _jsonable(row.get(field, "")) for field in fields})


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _num(value: Any, default: float = 0.0) -> float:
    try:
        if value in ("", None):
            return float(default)
        out = float(value)
    except Exception:
        return float(default)
    return out if math.isfinite(out) else float(default)


def _bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


class DSU:
    def __init__(self, ids: list[str]) -> None:
        self.parent = {item: item for item in ids}
        self.size = {item: 1 for item in ids}

    def find(self, item: str) -> str:
        parent = self.parent[item]
        if parent != item:
            self.parent[item] = self.find(parent)
        return self.parent[item]

    def union(self, a: str, b: str) -> bool:
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return False
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]
        return True


def _base_rows() -> list[dict[str, Any]]:
    rows = [
        dict(row)
        for row in _read_csv(PHASE10O_DIR / "mv_object_frame_mask_rows.csv")
        if row.get("variant_id") == BASE_VARIANT
    ]
    if not rows:
        raise RuntimeError(f"missing {BASE_VARIANT}")
    return rows


def _phase10p_edges() -> list[dict[str, Any]]:
    rows = [
        dict(row)
        for row in _read_csv(PHASE10P_DIR / "local2history_merge_rows.csv")
        if row.get("variant_id") == P10P_BEST_VARIANT
    ]
    if not rows:
        raise RuntimeError(f"missing Phase10P best merge rows for {P10P_BEST_VARIANT}")
    return rows


def _d4rt_edges() -> tuple[list[dict[str, Any]], dict[tuple[str, str, str], dict[str, dict[str, Any]]]]:
    rows = [
        dict(row)
        for row in _read_csv(PHASE10Y_DIR / "local2history_candidate_rows.csv")
        if row.get("anchor_family") in {"real_R20", "real_R40"}
    ]
    best_by_pair: dict[tuple[str, str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        key = (row.get("scene_id", ""), row.get("mv_object_id_a", ""), row.get("mv_object_id_b", ""))
        family = row.get("anchor_family", "")
        prev = best_by_pair[key].get(family)
        if prev is None or _num(row.get("object_anchor_overlap")) > _num(prev.get("object_anchor_overlap")):
            best_by_pair[key][family] = row
    return rows, best_by_pair


def _component_stats(mapping: dict[str, str]) -> dict[str, Any]:
    comps: dict[str, int] = defaultdict(int)
    for root in mapping.values():
        comps[root] += 1
    original = len(mapping)
    scene_objects = len(comps)
    merged = original - scene_objects
    large = sum(1 for size in comps.values() if size > 3)
    return {
        "scene_object_count": scene_objects,
        "history_merge_count": merged,
        "duplicate_scene_object_count": merged,
        "fragmentation_rate_proxy": float(scene_objects / original) if original else 0.0,
        "overmerge_rate_proxy_large_component_gt3": float(large / scene_objects) if scene_objects else 0.0,
        "max_component_size": max(comps.values()) if comps else 0,
    }


def _edge_score(row: dict[str, Any]) -> tuple[float, float]:
    if row.get("edge_source") == "d4rt":
        return _num(row.get("object_anchor_overlap")), _num(row.get("shared_anchor_count"))
    return _num(row.get("affinity")), _num(row.get("semantic_cosine"))


def _mapping_from_edges(ids: list[str], edges: list[dict[str, Any]], *, variant_id: str, policy: str) -> tuple[dict[str, str], list[dict[str, Any]]]:
    dsu = DSU(ids)
    used_left: set[tuple[str, str, str]] = set()
    used_right: set[tuple[str, str, str]] = set()
    accepted: list[dict[str, Any]] = []
    for row in sorted(edges, key=_edge_score, reverse=True):
        a = str(row.get("mv_object_id_a", ""))
        b = str(row.get("mv_object_id_b", ""))
        if a not in dsu.parent or b not in dsu.parent:
            continue
        left_key = (str(row.get("scene_id", "")), str(row.get("left_chunk_id", "")), a)
        right_key = (str(row.get("scene_id", "")), str(row.get("right_chunk_id", "")), b)
        if left_key in used_left or right_key in used_right:
            continue
        if not dsu.union(a, b):
            continue
        used_left.add(left_key)
        used_right.add(right_key)
        new = dict(row)
        new["schema_version"] = "stream4d_v99_phase10z_local2history_merge_v1"
        new["phase_id"] = "v99_phase10z_d4rt_verifier_semantic_scene_repair"
        new["variant_id"] = variant_id
        new["merge_policy"] = policy
        accepted.append(new)
    return {oid: f"{variant_id}:{dsu.find(oid)}" for oid in ids}, accepted


def _apply_mapping(rows: list[dict[str, Any]], *, variant_id: str, mapping: dict[str, str], policy: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        oid = str(row["mv_object_id"])
        new = dict(row)
        new["variant_id"] = variant_id
        new["variant"] = variant_id
        new["phase10z_parent_mv_object_id"] = oid
        new["mv_object_id"] = mapping.get(oid, f"{variant_id}:{oid}")
        new["object_id"] = new["mv_object_id"]
        new["object_id_policy"] = policy
        new["score_scope"] = "current_chunk_score_scene_stitched_identity"
        new["score_policy"] = str(row.get("score_policy", "")) + "__phase10z_d4rt_verifier_semantic_repair"
        new["uses_gt_for_prediction"] = False
        new["uses_future"] = False
        out.append(new)
    return out


def _aggregate_decoupled(variant_id: str, local_rows: list[dict[str, Any]], scene_rows: list[dict[str, Any]], stats: dict[str, Any]) -> dict[str, Any]:
    local_agg = p1._aggregate_metrics(local_rows)[0]
    scene_agg = p1._aggregate_metrics(scene_rows)[0]
    row = dict(local_agg)
    for key, value in scene_agg.items():
        if key.endswith("_scene"):
            row[key] = value
    row["variant_id"] = variant_id
    row["metric_composition"] = "local_from_phase10o_primary_scene_from_phase10z_stitched_ids"
    row.update(stats)
    return row


def _eval_worker(task: tuple[str, list[dict[str, Any]], dict[str, Any], str, int]) -> tuple[str, list[dict[str, Any]], list[dict[str, Any]]]:
    variant_id, rows, eval_scope, backend, device_id = task
    if backend == "cupy":
        from tools.v99_cupy_sparse_iou import CuPySparseSceneIoU

        class _DeviceCuPySparseSceneIoU(CuPySparseSceneIoU):
            def __init__(self) -> None:
                super().__init__(device_id=device_id)

        p1.SparseSceneIoU = _DeviceCuPySparseSceneIoU
    elif backend != "cpu":
        raise ValueError(f"unknown eval backend: {backend}")
    metric_rows, frame_rows = p1._evaluate_variant(variant_id, rows, eval_scope)
    return variant_id, metric_rows, frame_rows


def _parallel_eval(tasks: list[tuple[str, list[dict[str, Any]], dict[str, Any], str, int]], *, worker_count: int) -> dict[str, tuple[list[dict[str, Any]], list[dict[str, Any]]]]:
    if worker_count <= 1 or len(tasks) <= 1:
        return {variant_id: (metrics, frames) for variant_id, metrics, frames in (_eval_worker(task) for task in tasks)}
    try:
        ctx = mp.get_context("fork")
    except ValueError:
        ctx = mp.get_context()
    with ctx.Pool(processes=min(worker_count, len(tasks))) as pool:
        results = pool.map(_eval_worker, tasks)
    return {variant_id: (metrics, frames) for variant_id, metrics, frames in results}


def _d4rt_edges_for(family: str, tau: float, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        if row.get("anchor_family") != family:
            continue
        if _num(row.get("object_anchor_overlap")) < tau:
            continue
        new = dict(row)
        new["edge_source"] = "d4rt"
        new["affinity"] = new.get("object_anchor_overlap", "0")
        new["candidate_family"] = f"d4rt_{family}_tau{tau:.2f}"
        out.append(new)
    return out


def _confirmed_semantic_edges(phase10p_edges: list[dict[str, Any]], d4rt_by_pair: dict[tuple[str, str, str], dict[str, dict[str, Any]]], *, family: str, tau: float, keep_exact: bool) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in phase10p_edges:
        is_exact = row.get("candidate_family") == "exact_overlap_frame_mask"
        key = (row.get("scene_id", ""), row.get("mv_object_id_a", ""), row.get("mv_object_id_b", ""))
        d4 = d4rt_by_pair.get(key, {}).get(family)
        d4_overlap = _num(d4.get("object_anchor_overlap")) if d4 is not None else 0.0
        keep = bool((keep_exact and is_exact) or d4_overlap >= tau)
        if keep:
            new = dict(row)
            new["edge_source"] = "semantic_confirmed_by_d4rt" if d4_overlap >= tau else "semantic_exact_kept"
            new["d4rt_anchor_family"] = family
            new["d4rt_object_anchor_overlap"] = d4_overlap
            new["d4rt_shared_anchor_count"] = d4.get("shared_anchor_count", "") if d4 is not None else ""
            out.append(new)
    return out


def main() -> int:
    global OUT_DIR
    started = datetime.now()
    eval_workers = max(1, int(os.environ.get("V99_EVAL_WORKERS", str(DEFAULT_EVAL_WORKERS))))
    eval_backend = os.environ.get("V99_EVAL_BACKEND", "cpu").strip().lower()
    if eval_backend not in {"cpu", "cupy"}:
        raise ValueError(f"V99_EVAL_BACKEND must be cpu or cupy, got {eval_backend!r}")
    if os.environ.get("V99_PHASE10Z_OUT_DIR"):
        OUT_DIR = Path(os.environ["V99_PHASE10Z_OUT_DIR"])
    elif eval_backend == "cupy":
        OUT_DIR = AUDIT_ROOT / "v99_phase10z_d4rt_verifier_semantic_scene_repair_cupy"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    p10k._patch_phase1_inputs()
    phase0 = json.loads((PHASE0_DIR / "summary.json").read_text(encoding="utf-8"))
    phase10p_summary = json.loads((PHASE10P_DIR / "summary.json").read_text(encoding="utf-8"))
    phase10y_summary = json.loads((PHASE10Y_DIR / "summary.json").read_text(encoding="utf-8"))

    scope = p10o._build_overlap3_scope()
    eval_scope = p10o._eval_scope_from_overlap(scope)
    visible_devices = [tok for tok in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if tok.strip()]
    cupy_device_count = max(1, int(os.environ.get("V99_EVAL_CUPY_DEVICE_COUNT", str(len(visible_devices) if visible_devices else 1))))
    base_rows = _base_rows()
    ids = sorted({str(row["mv_object_id"]) for row in base_rows})
    p_edges = _phase10p_edges()
    d4_rows, d4_by_pair = _d4rt_edges()
    base_metric_rows, base_frame_rows = p1._evaluate_variant(BASE_VARIANT, base_rows, eval_scope)
    base_agg = p1._aggregate_metrics(base_metric_rows)[0]
    base_agg["metric_composition"] = "phase10o_primary_chunk_scoped_ids"
    base_agg.update(_component_stats({oid: f"{BASE_VARIANT}:{oid}" for oid in ids}))
    base_agg["accepted_history_merge_edge_count"] = 0
    base_agg["variant_family"] = "base"

    variant_specs: list[dict[str, Any]] = [
        {
            "variant_id": "Z0_phase10p_best_semantic_replay",
            "variant_family": "semantic_replay",
            "policy": "phase10p_best_semantic_edges",
            "edges": [dict(row, edge_source="phase10p_semantic") for row in p_edges],
        },
        {
            "variant_id": "Z1_d4rt_R40_tau0p20_only",
            "variant_family": "d4rt_high_tau_only",
            "policy": "d4rt_real_R40_tau0p20_only",
            "edges": _d4rt_edges_for("real_R40", 0.20, d4_rows),
        },
        {
            "variant_id": "Z2_d4rt_R40_tau0p40_only",
            "variant_family": "d4rt_high_tau_only",
            "policy": "d4rt_real_R40_tau0p40_only",
            "edges": _d4rt_edges_for("real_R40", 0.40, d4_rows),
        },
        {
            "variant_id": "Z3_d4rt_R20_tau0p20_only",
            "variant_family": "d4rt_high_tau_only",
            "policy": "d4rt_real_R20_tau0p20_only",
            "edges": _d4rt_edges_for("real_R20", 0.20, d4_rows),
        },
        {
            "variant_id": "Z4_semantic_exact_plus_R40confirm_tau0p02",
            "variant_family": "d4rt_verifier",
            "policy": "keep_exact_semantic_and_semantic_edges_confirmed_by_d4rt_R40_tau0p02",
            "edges": _confirmed_semantic_edges(p_edges, d4_by_pair, family="real_R40", tau=0.02, keep_exact=True),
        },
        {
            "variant_id": "Z5_semantic_exact_plus_R40confirm_tau0p04",
            "variant_family": "d4rt_verifier",
            "policy": "keep_exact_semantic_and_semantic_edges_confirmed_by_d4rt_R40_tau0p04",
            "edges": _confirmed_semantic_edges(p_edges, d4_by_pair, family="real_R40", tau=0.04, keep_exact=True),
        },
        {
            "variant_id": "Z6_semantic_plus_d4rt_R40_tau0p40",
            "variant_family": "semantic_plus_d4rt",
            "policy": "phase10p_best_edges_plus_d4rt_R40_tau0p40",
            "edges": [dict(row, edge_source="phase10p_semantic") for row in p_edges] + _d4rt_edges_for("real_R40", 0.40, d4_rows),
        },
        {
            "variant_id": "Z7_semantic_plus_d4rt_R20_tau0p20",
            "variant_family": "semantic_plus_d4rt",
            "policy": "phase10p_best_edges_plus_d4rt_R20_tau0p20",
            "edges": [dict(row, edge_source="phase10p_semantic") for row in p_edges] + _d4rt_edges_for("real_R20", 0.20, d4_rows),
        },
    ]

    metric_rows: list[dict[str, Any]] = [base_agg]
    same_identity_metric_rows: list[dict[str, Any]] = [base_agg]
    scene_metric_rows: list[dict[str, Any]] = list(base_metric_rows)
    frame_rows: list[dict[str, Any]] = list(base_frame_rows)
    merge_rows: list[dict[str, Any]] = []
    config_rows: list[dict[str, Any]] = []
    prepared_variants: list[dict[str, Any]] = []

    for spec in variant_specs:
        variant_id = str(spec["variant_id"])
        edges = list(spec["edges"])
        mapping, accepted = _mapping_from_edges(ids, edges, variant_id=variant_id, policy=str(spec["policy"]))
        rows = _apply_mapping(base_rows, variant_id=variant_id, mapping=mapping, policy=str(spec["policy"]))
        stats = _component_stats(mapping)
        stats["accepted_history_merge_edge_count"] = len(accepted)
        stats["candidate_edge_count"] = len(edges)
        stats["variant_family"] = spec["variant_family"]
        prepared_variants.append(
            {
                "variant_id": variant_id,
                "rows": rows,
                "stats": stats,
                "accepted": accepted,
                "config": {
                    "schema_version": "stream4d_v99_phase10z_variant_config_v1",
                    "phase_id": "v99_phase10z_d4rt_verifier_semantic_scene_repair",
                    "variant_id": variant_id,
                    "variant_family": spec["variant_family"],
                    "policy": spec["policy"],
                    "candidate_edge_count": len(edges),
                    "accepted_history_merge_edge_count": len(accepted),
                    "method_chunk_size": CHUNK_SIZE,
                    "method_chunk_overlap": OVERLAP,
                    "cross_model_geometry_edge_used": False,
                    "da3_d4rt_sim3_alignment_used": bool(phase10y_summary.get("da3_d4rt_sim3_alignment_used", False)),
                    "uses_gt_for_prediction": False,
                    "uses_future": False,
                },
            }
        )

    eval_tasks = [
        (str(item["variant_id"]), item["rows"], eval_scope, eval_backend, idx % cupy_device_count)
        for idx, item in enumerate(prepared_variants)
    ]
    print(
        json.dumps(
            {
                "event": "parallel_eval_start",
                "variant_count": len(eval_tasks),
                "eval_workers": min(eval_workers, len(eval_tasks)) if eval_tasks else 0,
                "backend": "multiprocessing_fork",
                "eval_backend": eval_backend,
                "cupy_visible_device_count": cupy_device_count if eval_backend == "cupy" else 0,
                "gpu_parallel_used": eval_backend == "cupy",
                "gpu_parallel_note": (
                    "CuPy backend moves SparseSceneIoU.add unique/count work to CUDA; PNG decode, raster assembly, and AP matching remain CPU."
                    if eval_backend == "cupy"
                    else "v65 AP evaluator is numpy/cpu/io bound in this implementation"
                ),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    eval_results = _parallel_eval(eval_tasks, worker_count=eval_workers)

    for item in prepared_variants:
        variant_id = str(item["variant_id"])
        stats = dict(item["stats"])
        accepted = list(item["accepted"])
        per_metric, frames = eval_results[variant_id]
        same_agg = p1._aggregate_metrics(per_metric)[0]
        same_agg["metric_composition"] = "local_and_scene_from_same_phase10z_stitched_ids"
        same_agg.update(stats)
        decoupled = _aggregate_decoupled(variant_id, base_metric_rows, per_metric, stats)
        metric_rows.append(decoupled)
        same_identity_metric_rows.append(same_agg)
        scene_metric_rows.extend(per_metric)
        frame_rows.extend(frames)
        merge_rows.extend(accepted)
        config_rows.append(dict(item["config"]))
        print(
            json.dumps(
                {
                    "event": "evaluated_variant",
                    "variant_id": variant_id,
                    "candidate_edge_count": len(edges),
                    "accepted_history_merge_edge_count": len(accepted),
                    "MV_AP_window": decoupled.get("MV_AP_window"),
                    "MV_AP_scene": decoupled.get("MV_AP_scene"),
                    "same_identity_MV_AP_window": same_agg.get("MV_AP_window"),
                    "same_identity_MV_AP_scene": same_agg.get("MV_AP_scene"),
                },
                sort_keys=True,
            ),
            flush=True,
        )

    real_rows = [row for row in metric_rows if row.get("variant_id") != BASE_VARIANT]
    best = max(real_rows, key=lambda row: (_num(row.get("MV_AP_scene")), _num(row.get("MV_AP50_scene")), _num(row.get("MV_AP_window"))))
    best_same = max([row for row in same_identity_metric_rows if row.get("variant_id") != BASE_VARIANT], key=lambda row: (_num(row.get("MV_AP_scene")), _num(row.get("MV_AP50_scene")), _num(row.get("MV_AP_window"))))
    holdout_scene_gate = float(phase0["F2_base_holdout_MV_AP_scene"]) + 0.010
    holdout_scene_ap50_gate = float(phase0["F2_base_holdout_MV_AP50_scene"]) + 0.015
    holdout_window_floor = float(phase0["F2_base_holdout_MV_AP_window"]) - 0.003
    scene_gate = (
        _num(best.get("MV_AP_scene")) >= holdout_scene_gate
        and _num(best.get("MV_AP50_scene")) >= holdout_scene_ap50_gate
        and _num(best.get("MV_AP_window")) >= holdout_window_floor
    )
    same_identity_window_floor_gate = _num(best_same.get("MV_AP_window")) >= holdout_window_floor
    safety_gate = (
        int(_num(best.get("same_frame_collision_count"), 1)) == 0
        and int(_num(best.get("missing_mask_raster_count"), 1)) == 0
        and not bool(scope.get("source_uses_future", False))
        and not bool(scope.get("source_uses_gt_for_prediction", False))
        and not bool(phase10y_summary.get("anchor_stats", {}).get("source_uses_future", False))
        and not bool(phase10y_summary.get("anchor_stats", {}).get("source_uses_gt_for_prediction", False))
    )
    metric_gate_pass = bool(scene_gate and same_identity_window_floor_gate and safety_gate)
    gate_rows = [
        {
            "gate_id": "scene_holdout_gate_vs_F2_holdout",
            "pass": scene_gate,
            "expected": f"MV_AP_scene>={holdout_scene_gate} and MV_AP50_scene>={holdout_scene_ap50_gate} and MV_AP_window>={holdout_window_floor}",
            "observed": f"{best['variant_id']} MV_AP_scene={best.get('MV_AP_scene')} MV_AP50_scene={best.get('MV_AP50_scene')} MV_AP_window={best.get('MV_AP_window')}",
            "severity": "scene_method_gate",
        },
        {
            "gate_id": "same_identity_window_floor",
            "pass": same_identity_window_floor_gate,
            "expected": f"same-identity MV_AP_window>={holdout_window_floor}",
            "observed": f"{best_same['variant_id']} MV_AP_window={best_same.get('MV_AP_window')} MV_AP_scene={best_same.get('MV_AP_scene')}",
            "severity": "diagnostic_consistency",
        },
        {
            "gate_id": "safety_no_gt_no_future_no_missing_no_collision",
            "pass": safety_gate,
            "expected": "uses_gt=false; uses_future=false; collisions=0; missing_mask=0",
            "observed": f"scope_uses_gt={scope.get('source_uses_gt_for_prediction')} scope_uses_future={scope.get('source_uses_future')} same_frame_collision={best.get('same_frame_collision_count')} missing_mask={best.get('missing_mask_raster_count')}",
            "severity": "safety",
        },
        {
            "gate_id": "formal_claim_allowed_after_cross_model_alignment",
            "pass": False,
            "expected": "DA3<->D4RT Sim3/scale alignment if mixed geometry is used",
            "observed": "this phase uses D4RT-only verifier/candidates; cross_model_geometry_edge_used=false",
            "severity": "formal_claim_blocker",
        },
    ]
    failure_rows = [
        {
            "failure_id": row["gate_id"],
            "severity": row["severity"],
            "expected": row["expected"],
            "observed": row["observed"],
            "repair_direction": "If D4RT verifier does not improve Phase10P, stop treating D4RT anchors as a main method component; next repair is DA3-D4RT scale/Sim3 alignment audit before any mixed geometry cue.",
        }
        for row in gate_rows
        if not bool(row["pass"])
    ]
    casebook_rows = [
        {
            "schema_version": "stream4d_v99_phase10z_casebook_v1",
            "phase_id": "v99_phase10z_d4rt_verifier_semantic_scene_repair",
            "rank": idx,
            "variant_id": row.get("variant_id"),
            "variant_family": row.get("variant_family"),
            "metric_composition": row.get("metric_composition"),
            "MV_AP_window": row.get("MV_AP_window"),
            "MV_AP50_window": row.get("MV_AP50_window"),
            "MV_AP_scene": row.get("MV_AP_scene"),
            "MV_AP50_scene": row.get("MV_AP50_scene"),
            "accepted_history_merge_edge_count": row.get("accepted_history_merge_edge_count"),
            "history_merge_count": row.get("history_merge_count"),
            "scene_object_count": row.get("scene_object_count"),
            "max_component_size": row.get("max_component_size"),
        }
        for idx, row in enumerate(sorted(metric_rows, key=lambda r: (_num(r.get("MV_AP_scene")), _num(r.get("MV_AP50_scene"))), reverse=True), start=1)
    ]
    summary = {
        "schema_version": "stream4d_v99_phase10z_d4rt_verifier_semantic_scene_repair_summary_v1",
        "phase_id": "v99_phase10z_d4rt_verifier_semantic_scene_repair",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "runtime_sec": (datetime.now() - started).total_seconds(),
        "decision": "GO_D4RT_VERIFIER_SEMANTIC_SCENE_REPAIRED_FORMAL_ALIGNMENT_REQUIRED" if metric_gate_pass else "NO_GO_D4RT_VERIFIER_SEMANTIC_SCENE_REPAIR",
        "metric_gate_pass": metric_gate_pass,
        "scene_gate_pass": bool(scene_gate),
        "same_identity_window_floor_gate_pass": bool(same_identity_window_floor_gate),
        "safety_gate_pass": bool(safety_gate),
        "formal_claim_allowed": False,
        "cross_model_geometry_edge_used": False,
        "da3_d4rt_sim3_alignment_used": bool(phase10y_summary.get("da3_d4rt_sim3_alignment_used", False)),
        "d4rt_candidate_source_summary": _rel(PHASE10Y_DIR / "summary.json"),
        "eval_parallel_backend": "multiprocessing_fork",
        "eval_parallel_workers": min(eval_workers, len(eval_tasks)) if eval_tasks else 0,
        "eval_backend": eval_backend,
        "cupy_visible_device_count": cupy_device_count if eval_backend == "cupy" else 0,
        "gpu_parallel_used": eval_backend == "cupy",
        "gpu_parallel_note": (
            "CuPy backend moves SparseSceneIoU.add unique/count work to CUDA; PNG decode, raster assembly, and AP matching remain CPU."
            if eval_backend == "cupy"
            else "Current v65 AP path is NumPy/CPU/I-O bound; GPU acceleration requires a separate torch/CUDA accumulator rewrite."
        ),
        "best_variant_id": best["variant_id"],
        "best_MV_AP_window": float(_num(best.get("MV_AP_window"))),
        "best_MV_AP50_window": float(_num(best.get("MV_AP50_window"))),
        "best_MV_AP_scene": float(_num(best.get("MV_AP_scene"))),
        "best_MV_AP50_scene": float(_num(best.get("MV_AP50_scene"))),
        "best_same_identity_variant_id": best_same["variant_id"],
        "best_same_identity_MV_AP_window": float(_num(best_same.get("MV_AP_window"))),
        "best_same_identity_MV_AP_scene": float(_num(best_same.get("MV_AP_scene"))),
        "base_MV_AP_window": float(_num(base_agg.get("MV_AP_window"))),
        "base_MV_AP_scene": float(_num(base_agg.get("MV_AP_scene"))),
        "phase10p_reported_best_variant_id": phase10p_summary.get("best_variant_id"),
        "phase10p_reported_best_MV_AP_scene": phase10p_summary.get("best_MV_AP_scene"),
        "phase10y_best_real_MV_AP_scene": phase10y_summary.get("best_real_MV_AP_scene"),
        "F2_base_holdout_MV_AP_window": float(phase0["F2_base_holdout_MV_AP_window"]),
        "F2_base_holdout_MV_AP_scene": float(phase0["F2_base_holdout_MV_AP_scene"]),
        "blocking_failure_count": len(failure_rows),
        "outputs": {
            "summary": _rel(OUT_DIR / "summary.json"),
            "variant_metric_rows": _rel(OUT_DIR / "variant_metric_rows.csv"),
            "same_identity_metric_rows": _rel(OUT_DIR / "same_identity_metric_rows.csv"),
            "variant_metric_scene_rows": _rel(OUT_DIR / "variant_metric_scene_rows.csv"),
            "variant_gate_rows": _rel(OUT_DIR / "variant_gate_rows.csv"),
            "variant_failure_rows": _rel(OUT_DIR / "variant_failure_rows.csv"),
            "variant_config_rows": _rel(OUT_DIR / "variant_config_rows.csv"),
            "local2history_merge_rows": _rel(OUT_DIR / "local2history_merge_rows.csv"),
            "casebook_rows": _rel(OUT_DIR / "casebook_rows.csv"),
            "frame_eval_rows": _rel(OUT_DIR / "frame_eval_rows.csv"),
        },
    }
    _write_csv(OUT_DIR / "variant_metric_rows.csv", metric_rows)
    _write_csv(OUT_DIR / "same_identity_metric_rows.csv", same_identity_metric_rows)
    _write_csv(OUT_DIR / "variant_metric_scene_rows.csv", scene_metric_rows)
    _write_csv(OUT_DIR / "variant_gate_rows.csv", gate_rows)
    _write_csv(OUT_DIR / "variant_failure_rows.csv", failure_rows)
    _write_csv(OUT_DIR / "variant_config_rows.csv", config_rows)
    _write_csv(OUT_DIR / "local2history_merge_rows.csv", merge_rows)
    _write_csv(OUT_DIR / "casebook_rows.csv", casebook_rows)
    _write_csv(OUT_DIR / "frame_eval_rows.csv", frame_rows)
    _write_json(OUT_DIR / "summary.json", summary)
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))
    return 0 if metric_gate_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
