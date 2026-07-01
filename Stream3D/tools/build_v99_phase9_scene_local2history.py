#!/usr/bin/env python3
"""Evaluate v99 Phase9 scene/local2history stitching from F2 rows."""

from __future__ import annotations

import csv
import json
import math
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
from tools import build_v99_phase4_f2_da3_link_verifier as p4  # noqa: E402


AUDIT_ROOT = STREAM3D_ROOT / "outputs/audit"
OUT_DIR = AUDIT_ROOT / "v99_phase9_scene_local2history"
PHASE0_DIR = AUDIT_ROOT / "v99_phase0_fact_lock"
PHASE2_DIR = AUDIT_ROOT / "v99_phase2_f2_strengthening"
PHASE5_DIR = AUDIT_ROOT / "v99_phase5_d4rt_anchor_verifier"
PHASE8_DIR = AUDIT_ROOT / "v99_phase8_fusion_matrix"


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
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys or ["schema_version"])
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _jsonable(row.get(key, "")) for key in keys})


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


def _normalize(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm <= 1e-12:
        return vec.astype(np.float32)
    return (vec / norm).astype(np.float32)


class DSU:
    def __init__(self, ids: list[str]) -> None:
        self.parent = {item: item for item in ids}
        self.size = {item: 1 for item in ids}

    def find(self, item: str) -> str:
        if self.parent[item] != item:
            self.parent[item] = self.find(self.parent[item])
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


def _phase2_best_rows() -> tuple[str, list[dict[str, Any]]]:
    summary = json.loads((PHASE2_DIR / "best_variant_summary.json").read_text(encoding="utf-8"))
    variant = str(summary["best_variant_id"])
    rows = [dict(row) for row in _read_csv(PHASE2_DIR / "mv_object_frame_mask_rows.csv") if row.get("variant_id") == variant]
    if not rows:
        raise RuntimeError(f"no Phase2 best rows for {variant}")
    return variant, rows


def _object_infos(rows: list[dict[str, Any]], scope: dict[str, Any]) -> dict[str, dict[str, Any]]:
    features, _tau = p1._load_radio_residual_features()
    infos: dict[str, dict[str, Any]] = defaultdict(lambda: {"rows": [], "frames": set(), "features": []})
    for row in rows:
        oid = str(row["mv_object_id"])
        scene = str(row["scene_id"])
        chunk = str(row["chunk_id"])
        frame = int(float(row["frame_id"]))
        mask = int(float(row["selected_mask_id"]))
        infos[oid]["rows"].append(row)
        infos[oid]["scene_id"] = scene
        infos[oid]["chunk_id"] = chunk
        infos[oid]["frames"].add(frame)
        feat = features.get((scene, frame, mask))
        if feat is not None:
            infos[oid]["features"].append(feat)
    for oid, info in infos.items():
        frames = sorted(info["frames"])
        feats = info["features"]
        info["first_frame"] = frames[0] if frames else -1
        info["last_frame"] = frames[-1] if frames else -1
        info["frame_count"] = len(frames)
        if feats:
            info["feature"] = _normalize(np.mean(np.stack(feats).astype(np.float32), axis=0))
        else:
            info["feature"] = None
    return dict(infos)


def _chunk_order(infos: dict[str, dict[str, Any]]) -> dict[str, list[str]]:
    by_scene: dict[str, set[str]] = defaultdict(set)
    for info in infos.values():
        by_scene[str(info["scene_id"])].add(str(info["chunk_id"]))
    return {scene: sorted(chunks) for scene, chunks in by_scene.items()}


def _object_geometry(infos: dict[str, dict[str, Any]], scope: dict[str, Any]) -> dict[str, dict[str, Any]]:
    da3_maps = p4._load_da3_maps()
    geom_cache: dict[tuple[str, int, int], dict[str, Any]] = {}
    out: dict[str, dict[str, Any]] = {}
    for oid, info in infos.items():
        rows = sorted(info["rows"], key=lambda row: int(float(row["frame_id"])))
        if not rows:
            continue
        sample_indices = sorted({0, len(rows) // 2, len(rows) - 1})
        centroids: list[np.ndarray] = []
        radii: list[float] = []
        confs: list[float] = []
        for idx in sample_indices:
            row = rows[idx]
            geom = p4._mask_geometry(
                str(row["scene_id"]),
                int(float(row["frame_id"])),
                int(float(row["selected_mask_id"])),
                da3_maps=da3_maps,
                mask_path_by_frame=scope["mask_path_by_frame"],
                cache=geom_cache,
            )
            if bool(geom.get("valid")):
                centroids.append(np.asarray(geom["centroid"], dtype=np.float32))
                radii.append(float(geom.get("radius", 0.0)))
                confs.append(float(geom.get("mean_conf", 0.0)))
        if centroids:
            out[oid] = {
                "valid": True,
                "centroid": np.mean(np.stack(centroids), axis=0).astype(np.float32),
                "radius": float(np.mean(radii)) if radii else 0.0,
                "mean_conf": float(np.mean(confs)) if confs else 0.0,
                "sample_count": len(centroids),
            }
        else:
            out[oid] = {
                "valid": False,
                "centroid": np.zeros(3, dtype=np.float32),
                "radius": 0.0,
                "mean_conf": 0.0,
                "sample_count": 0,
            }
    return out


def _adjacent_pairs(infos: dict[str, dict[str, Any]]) -> list[tuple[str, str, str, str]]:
    order = _chunk_order(infos)
    by_scene_chunk: dict[tuple[str, str], list[str]] = defaultdict(list)
    for oid, info in infos.items():
        by_scene_chunk[(str(info["scene_id"]), str(info["chunk_id"]))].append(oid)
    pairs: list[tuple[str, str, str, str]] = []
    for scene, chunks in order.items():
        for left, right in zip(chunks[:-1], chunks[1:]):
            for a in sorted(by_scene_chunk[(scene, left)]):
                for b in sorted(by_scene_chunk[(scene, right)]):
                    pairs.append((scene, left, a, b))
    return pairs


def _semantic_candidates(infos: dict[str, dict[str, Any]], *, tau: float) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for scene, left_chunk, a, b in _adjacent_pairs(infos):
        fa = infos[a].get("feature")
        fb = infos[b].get("feature")
        if fa is None or fb is None:
            continue
        sim = float(np.dot(fa, fb))
        if sim >= tau:
            rows.append(
                {
                    "scene_id": scene,
                    "left_chunk_id": left_chunk,
                    "right_chunk_id": infos[b]["chunk_id"],
                    "mv_object_id_a": a,
                    "mv_object_id_b": b,
                    "semantic_cosine": sim,
                    "affinity": sim,
                    "candidate_family": "semantic_residual",
                    "uses_gt_for_prediction": False,
                    "uses_future": False,
                }
            )
    return rows


def _semantic_da3_candidates(
    infos: dict[str, dict[str, Any]],
    geometry: dict[str, dict[str, Any]],
    *,
    semantic_tau: float,
    max_distance: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for scene, left_chunk, a, b in _adjacent_pairs(infos):
        fa = infos[a].get("feature")
        fb = infos[b].get("feature")
        ga = geometry.get(a, {})
        gb = geometry.get(b, {})
        if fa is None or fb is None or not ga.get("valid") or not gb.get("valid"):
            continue
        sim = float(np.dot(fa, fb))
        dist = float(np.linalg.norm(np.asarray(ga["centroid"]) - np.asarray(gb["centroid"])))
        if sim >= semantic_tau and dist <= max_distance:
            geo_aff = max(0.0, 1.0 - dist / max(max_distance, 1e-6))
            rows.append(
                {
                    "scene_id": scene,
                    "left_chunk_id": left_chunk,
                    "right_chunk_id": infos[b]["chunk_id"],
                    "mv_object_id_a": a,
                    "mv_object_id_b": b,
                    "semantic_cosine": sim,
                    "da3_centroid_distance": dist,
                    "da3_affinity": geo_aff,
                    "affinity": 0.7 * sim + 0.3 * geo_aff,
                    "candidate_family": "semantic_plus_DA3_centroid",
                    "uses_gt_for_prediction": False,
                    "uses_future": False,
                }
            )
    return rows


def _d4rt_candidates(*, tau: float) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in _read_csv(PHASE5_DIR / "local2history_merge_rows.csv"):
        overlap = _num(row.get("object_anchor_overlap"))
        if overlap >= tau:
            rows.append(
                {
                    "scene_id": row.get("scene_id"),
                    "left_chunk_id": row.get("left_chunk_id"),
                    "right_chunk_id": row.get("right_chunk_id"),
                    "mv_object_id_a": row.get("mv_object_id_a"),
                    "mv_object_id_b": row.get("mv_object_id_b"),
                    "object_anchor_overlap": overlap,
                    "shared_anchor_count": int(_num(row.get("shared_anchor_count"))),
                    "affinity": overlap,
                    "candidate_family": "D4RT_anchor_local2history",
                    "uses_gt_for_prediction": False,
                    "uses_future": False,
                }
            )
    return rows


def _one_to_one_unions(ids: list[str], candidates: list[dict[str, Any]], *, variant_id: str) -> tuple[dict[str, str], list[dict[str, Any]]]:
    dsu = DSU(ids)
    used_left: set[tuple[str, str, str]] = set()
    used_right: set[tuple[str, str, str]] = set()
    accepted: list[dict[str, Any]] = []
    for row in sorted(candidates, key=lambda item: (_num(item.get("affinity")), str(item.get("mv_object_id_a")), str(item.get("mv_object_id_b"))), reverse=True):
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
            new["schema_version"] = "stream4d_v99_phase9_local2history_merge_v1"
            new["phase_id"] = "v99_phase9_scene_local2history"
            new["variant_id"] = variant_id
            new["merge_policy"] = "adjacent_chunk_one_to_one_greedy"
            accepted.append(new)
            used_left.add(left_key)
            used_right.add(right_key)
    mapping = {oid: f"{variant_id}:{dsu.find(oid)}" for oid in ids}
    return mapping, accepted


def _apply_mapping(rows: list[dict[str, Any]], *, variant_id: str, mapping: dict[str, str], policy: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        oid = str(row["mv_object_id"])
        new = dict(row)
        new["variant_id"] = variant_id
        new["phase9_parent_mv_object_id"] = oid
        new["mv_object_id"] = mapping.get(oid, f"{variant_id}:{oid}")
        new["object_id"] = new["mv_object_id"]
        new["object_id_policy"] = policy
        new["score_policy"] = str(row.get("score_policy", "")) + "__phase9_scene_stitch"
        new["score_scope"] = "current_chunk_score_scene_stitched_identity"
        new["uses_gt_for_prediction"] = False
        new["uses_future"] = False
        out.append(new)
    return out


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
        "history_split_count": 0,
        "duplicate_scene_object_count": merged,
        "fragmentation_rate_proxy": float(scene_objects / original) if original else 0.0,
        "overmerge_rate_proxy_large_component_gt3": float(large / scene_objects) if scene_objects else 0.0,
        "max_component_size": max(comps.values()) if comps else 0,
    }


def _evaluate_variant(variant_id: str, rows: list[dict[str, Any]], scope: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    metric_rows, frame_rows = p1._evaluate_variant(variant_id, rows, scope)
    aggregate = p1._aggregate_metrics(metric_rows)
    if len(aggregate) != 1:
        raise RuntimeError(f"expected one aggregate metric row for {variant_id}, got {len(aggregate)}")
    return aggregate[0], metric_rows, frame_rows


def main() -> int:
    started = datetime.now()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    phase0 = json.loads((PHASE0_DIR / "summary.json").read_text(encoding="utf-8"))
    phase8 = json.loads((PHASE8_DIR / "summary.json").read_text(encoding="utf-8"))
    parent_variant, parent_rows = _phase2_best_rows()
    scope = p1._load_source_scope()
    infos = _object_infos(parent_rows, scope)
    ids = sorted(infos)
    geometry = _object_geometry(infos, scope)

    by_scene_chunk_frames: dict[tuple[str, str], set[int]] = defaultdict(set)
    for row in parent_rows:
        by_scene_chunk_frames[(str(row["scene_id"]), str(row["chunk_id"]))].add(int(float(row["frame_id"])))
    shared_frame_pair_count = 0
    for scene, chunks in _chunk_order(infos).items():
        for left, right in zip(chunks[:-1], chunks[1:]):
            shared_frame_pair_count += len(by_scene_chunk_frames[(scene, left)] & by_scene_chunk_frames[(scene, right)])

    variant_specs: list[dict[str, Any]] = [
        {
            "variant_id": "H0_F2_window_fragmented",
            "family": "baseline",
            "policy": "no_scene_stitch",
            "candidates": [],
            "notes": "Phase2 best rows without scene stitching.",
        },
        {
            "variant_id": "H1_F2_overlap_tube_stitch",
            "family": "F2_overlap_tube",
            "policy": "overlap_tube_noop_no_shared_sampled_frames",
            "candidates": [],
            "notes": f"No shared sampled frames were found across adjacent chunks; shared_frame_pair_count={shared_frame_pair_count}.",
        },
        {
            "variant_id": "H2_F2_semantic_residual_tau0p95",
            "family": "F2_semantic_residual",
            "policy": "semantic_adjacent_chunk_one_to_one_tau0.95",
            "candidates": _semantic_candidates(infos, tau=0.95),
            "notes": "Adjacent chunk one-to-one greedy semantic residual stitch.",
        },
        {
            "variant_id": "H2_F2_semantic_residual_tau0p90",
            "family": "F2_semantic_residual",
            "policy": "semantic_adjacent_chunk_one_to_one_tau0.90",
            "candidates": _semantic_candidates(infos, tau=0.90),
            "notes": "Lower semantic threshold repair variant.",
        },
        {
            "variant_id": "H3_H2_plus_DA3_overlap_geo0p25",
            "family": "F2_semantic_plus_DA3",
            "policy": "semantic_tau0.90_da3_centroid_dist_le0.25_one_to_one",
            "candidates": _semantic_da3_candidates(infos, geometry, semantic_tau=0.90, max_distance=0.25),
            "notes": "DA3 geometry-constrained semantic stitch; uses centroids from verified official DA3 artifacts.",
        },
        {
            "variant_id": "H3_H2_plus_DA3_overlap_geo0p50",
            "family": "F2_semantic_plus_DA3",
            "policy": "semantic_tau0.90_da3_centroid_dist_le0.50_one_to_one",
            "candidates": _semantic_da3_candidates(infos, geometry, semantic_tau=0.90, max_distance=0.50),
            "notes": "Looser DA3 distance repair variant.",
        },
        {
            "variant_id": "H4_H2_plus_D4RT_anchor_tau0p30",
            "family": "F2_plus_D4RT_local2history",
            "policy": "d4rt_anchor_overlap_tau0.30_one_to_one",
            "candidates": _d4rt_candidates(tau=0.30),
            "notes": "D4RT local2history repair using one-to-one greedy instead of many-to-many union.",
        },
        {
            "variant_id": "H4_H2_plus_D4RT_anchor_tau0p50",
            "family": "F2_plus_D4RT_local2history",
            "policy": "d4rt_anchor_overlap_tau0.50_one_to_one",
            "candidates": _d4rt_candidates(tau=0.50),
            "notes": "Stricter D4RT local2history repair.",
        },
    ]
    h5_candidates = (
        _semantic_da3_candidates(infos, geometry, semantic_tau=0.90, max_distance=0.50)
        + _d4rt_candidates(tau=0.30)
    )
    variant_specs.append(
        {
            "variant_id": "H5_H2_plus_DA3_plus_D4RT",
            "family": "F2_semantic_plus_DA3_plus_D4RT",
            "policy": "semantic_da3_or_d4rt_one_to_one",
            "candidates": h5_candidates,
            "notes": "Union of H3 geo0.50 and H4 tau0.30 candidates under one-to-one greedy.",
        }
    )

    all_variant_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    scene_metric_rows: list[dict[str, Any]] = []
    frame_rows: list[dict[str, Any]] = []
    merge_rows: list[dict[str, Any]] = []
    config_rows: list[dict[str, Any]] = []
    for spec in variant_specs:
        variant_id = str(spec["variant_id"])
        if variant_id == "H0_F2_window_fragmented":
            mapping = {oid: f"{variant_id}:{oid}" for oid in ids}
            accepted: list[dict[str, Any]] = []
        elif variant_id == "H1_F2_overlap_tube_stitch":
            mapping = {oid: f"{variant_id}:{oid}" for oid in ids}
            accepted = []
        else:
            mapping, accepted = _one_to_one_unions(ids, list(spec["candidates"]), variant_id=variant_id)
        rows = _apply_mapping(parent_rows, variant_id=variant_id, mapping=mapping, policy=str(spec["policy"]))
        stats = _component_stats(mapping)
        aggregate, per_scene, frames = _evaluate_variant(variant_id, rows, scope)
        aggregate["phase_id"] = "v99_phase9_scene_local2history"
        aggregate["family"] = spec["family"]
        aggregate["history_candidate_count"] = len(spec["candidates"])
        aggregate["accepted_history_merge_edge_count"] = len(accepted)
        aggregate.update(stats)
        metric_rows.append(aggregate)
        scene_metric_rows.extend(per_scene)
        frame_rows.extend(frames)
        merge_rows.extend(accepted)
        all_variant_rows.extend(rows)
        config_rows.append(
            {
                "schema_version": "stream4d_v99_phase9_variant_config_v1",
                "phase_id": "v99_phase9_scene_local2history",
                "variant_id": variant_id,
                "family": spec["family"],
                "policy": spec["policy"],
                "parent_variant": parent_variant,
                "candidate_count": len(spec["candidates"]),
                "accepted_history_merge_edge_count": len(accepted),
                "shared_frame_pair_count": shared_frame_pair_count,
                "uses_gt_for_prediction": False,
                "uses_future": False,
                "notes": spec["notes"],
            }
        )

    best = max(metric_rows, key=lambda row: (_num(row.get("MV_AP_scene")), _num(row.get("MV_AP50_scene")), _num(row.get("MV_AP_window"))))
    best_window = max(metric_rows, key=lambda row: (_num(row.get("MV_AP_window")), _num(row.get("MV_AP_scene"))))
    h0 = next(row for row in metric_rows if row["variant_id"] == "H0_F2_window_fragmented")
    f2_base_scene = float(phase0["F2_base_full_dev_MV_AP_scene"])
    f2_base_ap50_scene = float(phase0["F2_base_full_dev_MV_AP50_scene"])
    f2_base_window = float(phase0["F2_base_full_dev_MV_AP_window"])
    scene_gate = bool(
        _num(best.get("MV_AP_scene")) >= f2_base_scene + 0.010
        and _num(best.get("MV_AP50_scene")) >= f2_base_ap50_scene + 0.015
        and _num(best.get("MV_AP_window")) >= f2_base_window - 0.003
    )
    no_local_collapse_gate = bool(_num(best.get("MV_AP_window")) >= f2_base_window - 0.003)
    gate_rows = [
        {
            "gate_id": "scene_MV_AP_scene_ge_F2_base_plus_0p010",
            "pass": _num(best.get("MV_AP_scene")) >= f2_base_scene + 0.010,
            "expected": f">={f2_base_scene + 0.010}",
            "observed": f"{best['variant_id']} MV_AP_scene={best.get('MV_AP_scene')}",
            "severity": "required_scene",
        },
        {
            "gate_id": "scene_MV_AP50_scene_ge_F2_base_plus_0p015",
            "pass": _num(best.get("MV_AP50_scene")) >= f2_base_ap50_scene + 0.015,
            "expected": f">={f2_base_ap50_scene + 0.015}",
            "observed": f"{best['variant_id']} MV_AP50_scene={best.get('MV_AP50_scene')}",
            "severity": "required_scene",
        },
        {
            "gate_id": "scene_candidate_MV_AP_window_not_lower_than_F2_base_minus_0p003",
            "pass": no_local_collapse_gate,
            "expected": f">={f2_base_window - 0.003}",
            "observed": f"{best['variant_id']} MV_AP_window={best.get('MV_AP_window')}",
            "severity": "required_scene",
        },
        {
            "gate_id": "safety_same_frame_collision_and_missing_mask",
            "pass": int(_num(best.get("same_frame_collision_count"))) == 0 and int(_num(best.get("missing_mask_raster_count"))) == 0,
            "expected": "same_frame_collision_count=0 and missing_mask_raster_count=0",
            "observed": f"same_frame_collision_count={best.get('same_frame_collision_count')}; missing_mask_raster_count={best.get('missing_mask_raster_count')}",
            "severity": "required",
        },
    ]
    failure_rows = [
        {
            "failure_id": row["gate_id"],
            "severity": row["severity"],
            "expected": row["expected"],
            "observed": row["observed"],
            "repair_direction": "Scene stitching failed Phase9 gate; keep local candidate and do not promote local2history unless a later pre-registered repair beats F2_base scene without local collapse.",
        }
        for row in gate_rows
        if not bool(row["pass"])
    ]
    decision = "PASS_SCENE_LOCAL2HISTORY_CANDIDATE" if scene_gate else "NO_GO_SCENE_LOCAL2HISTORY_KEEP_LOCAL_CANDIDATE"
    summary = {
        "schema_version": "stream4d_v99_phase9_scene_local2history_summary_v1",
        "phase_id": "v99_phase9_scene_local2history",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "runtime_sec": (datetime.now() - started).total_seconds(),
        "decision": decision,
        "phase9_pass": scene_gate,
        "parent_variant": parent_variant,
        "phase8_best_local_variant": phase8.get("best_real_variant"),
        "shared_frame_pair_count": shared_frame_pair_count,
        "best_scene_variant": best["variant_id"],
        "best_scene_MV_AP_window": float(_num(best.get("MV_AP_window"))),
        "best_scene_MV_AP50_window": float(_num(best.get("MV_AP50_window"))),
        "best_scene_MV_AP_scene": float(_num(best.get("MV_AP_scene"))),
        "best_scene_MV_AP50_scene": float(_num(best.get("MV_AP50_scene"))),
        "best_scene_history_merge_count": int(_num(best.get("history_merge_count"))),
        "best_scene_overmerge_rate_proxy_large_component_gt3": float(_num(best.get("overmerge_rate_proxy_large_component_gt3"))),
        "best_window_variant": best_window["variant_id"],
        "best_window_MV_AP_window": float(_num(best_window.get("MV_AP_window"))),
        "H0_MV_AP_window": float(_num(h0.get("MV_AP_window"))),
        "H0_MV_AP_scene": float(_num(h0.get("MV_AP_scene"))),
        "F2_base_MV_AP_window": f2_base_window,
        "F2_base_MV_AP_scene": f2_base_scene,
        "scene_gate_pass": scene_gate,
        "no_local_collapse_gate_pass": no_local_collapse_gate,
        "outputs": {
            "variant_config_rows": _rel(OUT_DIR / "variant_config_rows.csv"),
            "variant_metric_rows": _rel(OUT_DIR / "variant_metric_rows.csv"),
            "variant_metric_scene_rows": _rel(OUT_DIR / "variant_metric_scene_rows.csv"),
            "variant_gate_rows": _rel(OUT_DIR / "variant_gate_rows.csv"),
            "variant_failure_rows": _rel(OUT_DIR / "variant_failure_rows.csv"),
            "local2history_merge_rows": _rel(OUT_DIR / "local2history_merge_rows.csv"),
            "mv_object_frame_mask_rows": _rel(OUT_DIR / "mv_object_frame_mask_rows.csv"),
            "frame_eval_rows": _rel(OUT_DIR / "frame_eval_rows.csv"),
            "summary": _rel(OUT_DIR / "summary.json"),
        },
    }
    _write_csv(OUT_DIR / "variant_config_rows.csv", config_rows)
    _write_csv(OUT_DIR / "variant_metric_rows.csv", metric_rows)
    _write_csv(OUT_DIR / "variant_metric_scene_rows.csv", scene_metric_rows)
    _write_csv(OUT_DIR / "variant_gate_rows.csv", gate_rows)
    _write_csv(OUT_DIR / "variant_failure_rows.csv", failure_rows)
    _write_csv(OUT_DIR / "local2history_merge_rows.csv", merge_rows)
    _write_csv(OUT_DIR / "mv_object_frame_mask_rows.csv", all_variant_rows)
    _write_csv(OUT_DIR / "frame_eval_rows.csv", frame_rows)
    _write_json(OUT_DIR / "summary.json", summary)
    _write_json(OUT_DIR / "best_variant_summary.json", summary)
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))
    return 0 if scene_gate else 2


if __name__ == "__main__":
    raise SystemExit(main())
