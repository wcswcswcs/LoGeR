#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd


STREAM3D_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = STREAM3D_ROOT.parent
if str(STREAM3D_ROOT) not in sys.path:
    sys.path.insert(0, str(STREAM3D_ROOT))

from tools.build_v103_phase6_mask_clustering_local_object_birth import (  # noqa: E402
    _evaluate_variant,
)
from tools.build_v103_phase6d_f2_skeleton_affinity_merge import (  # noqa: E402
    DEFAULT_PHASE2_SCENE0011,
    DEFAULT_PHASE2_SCENE0050,
    UnionFind,
    _adapt_f2_rows,
    _broad_support_risk,
    _load_phase5_scene,
    _object_tables,
    _specific_conflict,
)
from tools.run_v65_scene_multiview_ap import _load_gt_2d  # noqa: E402


AUDIT_ROOT = STREAM3D_ROOT / "outputs" / "audit"
PHASE_ID = "v103_r7_phase4_skeleton_confirmed_support"
DEFAULT_OUT = AUDIT_ROOT / "v103_r7_phase4_skeleton_confirmed_support"
DEFAULT_R7_PHASE1_ROOT = AUDIT_ROOT / "v103_r7_phase1_edge_attribution_exact"
DEFAULT_F2_ROOT = AUDIT_ROOT / "v100_phase2c_overlap3_local_repair"
DEFAULT_PHASE5_ROOT = AUDIT_ROOT / "v103_supp_r6_phase2_support_conditioned_local_ap/phase5_like_features"
DEFAULT_SUPPORT_FEATURE = DEFAULT_PHASE5_ROOT / "R6F2_support010_specificity_semantic"
DEFAULT_SCENE0011_PHASE2 = AUDIT_ROOT / "v103_phase2_stratified_q5c_objlike16384_scene0011_c0001_cap24576_qchunk16384_d4rt48mix_maskbalanced8"
DEFAULT_SCENE0050_PHASE2 = AUDIT_ROOT / "v103_phase2_stratified_q5c_objlike16384_scene0050_c0001_cap24576_qchunk16384_d4rt48mix_maskbalanced8"


VARIANTS = [
    {
        "variant_id": "R7SS1_skeleton_support_confirm_tauS070",
        "support_threshold": 0.70,
        "max_temporal_gap": 10,
        "topk_per_object": 1,
        "broad_support_veto": True,
    },
    {
        "variant_id": "R7SS2_skeleton_support_confirm_tauS060",
        "support_threshold": 0.60,
        "max_temporal_gap": 10,
        "topk_per_object": 1,
        "broad_support_veto": True,
    },
    {
        "variant_id": "R7SS3_skeleton_support_confirm_semantic_gate",
        "support_threshold": 0.60,
        "max_temporal_gap": 10,
        "topk_per_object": 1,
        "broad_support_veto": True,
        "semantic_gate": True,
    },
    {
        "variant_id": "R7SS4_skeleton_support_confirm_veto_attenuated",
        "support_threshold": 0.55,
        "max_temporal_gap": 10,
        "topk_per_object": 1,
        "broad_support_veto": True,
        "broad_support_min_support_mean": 1500.0,
    },
    {
        "variant_id": "R7SS5_skeleton_support_confirm_temporal_neighbor_only",
        "support_threshold": 0.60,
        "max_temporal_gap": 5,
        "topk_per_object": 1,
        "broad_support_veto": True,
        "temporal_neighbor_only": True,
    },
]


def _project(path: Path | str) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    if p.parts and p.parts[0] == "Stream3D":
        return REPO_ROOT / p
    return STREAM3D_ROOT / p


def _rel(path: Path | str) -> str:
    p = Path(path)
    try:
        return p.resolve().relative_to(REPO_ROOT).as_posix()
    except Exception:
        return p.as_posix()


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


def _sha256(path: Path) -> str:
    if not path.exists() or not path.is_file():
        return ""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


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


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _num(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        out = float(value)
        return out if np.isfinite(out) else default
    except Exception:
        return default


def _artifact_row(role: str, path: Path, *, required: bool = True, note: str = "") -> dict[str, Any]:
    exists = path.exists()
    row_count: int | str = ""
    if exists and path.is_file() and path.suffix.lower() in {".csv", ".parquet"}:
        try:
            if path.suffix.lower() == ".csv":
                row_count = max(sum(1 for _ in path.open("r", encoding="utf-8")) - 1, 0)
            else:
                row_count = int(pd.read_parquet(path).shape[0])
        except Exception:
            row_count = ""
    return {
        "schema_version": "stream4d_v103_r7_phase4_artifact_row_v1",
        "phase_id": PHASE_ID,
        "artifact_role": role,
        "path": _rel(path),
        "exists": bool(exists),
        "required": bool(required),
        "is_file": path.is_file() if exists else False,
        "is_dir": path.is_dir() if exists else False,
        "size_bytes": path.stat().st_size if exists and path.is_file() else "",
        "sha256": _sha256(path),
        "row_count": row_count,
        "note": note,
        "uses_gt_for_prediction": False,
        "uses_future": False,
    }


def _gate_row(name: str, passed: bool, observed: Any, required: Any, repair: str) -> dict[str, Any]:
    return {
        "schema_version": "stream4d_v103_r7_phase4_gate_row_v1",
        "phase_id": PHASE_ID,
        "gate_name": name,
        "pass": bool(passed),
        "observed": observed,
        "required": required,
        "repair_direction": repair,
        "uses_gt_for_prediction": False,
        "uses_future": False,
    }


def _failure_row(failure_id: str, detail: Any, repair: str, severity: str = "blocking") -> dict[str, Any]:
    return {
        "schema_version": "stream4d_v103_r7_phase4_failure_row_v1",
        "phase_id": PHASE_ID,
        "failure_id": failure_id,
        "severity": severity,
        "detail": json.dumps(_jsonable(detail), sort_keys=True) if isinstance(detail, (dict, list, tuple)) else detail,
        "repair_direction": repair,
        "uses_gt_for_prediction": False,
        "uses_future": False,
    }


def _label_path(mask_root: str, frame_id: int) -> Path:
    return _project(mask_root) / f"{int(frame_id)}.png"


def _load_label(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(path)
    if img.ndim == 3:
        img = img[..., 0]
    return np.asarray(img, dtype=np.int64)


def _object_extent(group: pd.DataFrame) -> tuple[int, int, int]:
    frames = sorted({int(v) for v in group["frame_id"].tolist()})
    if not frames:
        return 0, 0, 0
    return min(frames), max(frames), len(frames)


def _temporal_gap(group_a: pd.DataFrame, group_b: pd.DataFrame) -> int:
    a0, a1, _ = _object_extent(group_a)
    b0, b1, _ = _object_extent(group_b)
    if a1 < b0:
        return int(b0 - a1)
    if b1 < a0:
        return int(a0 - b1)
    return 0


def _skeleton_score(group_a: pd.DataFrame, group_b: pd.DataFrame, max_gap: int) -> float:
    gap = _temporal_gap(group_a, group_b)
    if gap > max_gap:
        return 0.0
    _, _, fa = _object_extent(group_a)
    _, _, fb = _object_extent(group_b)
    overlap_frames = len(set(group_a["frame_id"].astype(int).tolist()).intersection(set(group_b["frame_id"].astype(int).tolist())))
    temporal = 1.0 - min(float(gap), float(max_gap)) / max(float(max_gap), 1.0)
    coverage = min(float(fa), float(fb)) / max(float(fa), float(fb), 1.0)
    overlap_bonus = min(0.25, 0.05 * float(overlap_frames))
    return float(0.55 * temporal + 0.35 * coverage + overlap_bonus)


def _semantic_gate_ok(group_a: pd.DataFrame, group_b: pd.DataFrame) -> bool:
    rates = []
    for group in [group_a, group_b]:
        broad = float(group["selected_mask_is_broad"].astype(bool).mean())
        obj = float(group["selected_mask_is_object_like"].astype(bool).mean())
        rates.append((broad, obj))
    return all(obj >= 0.5 and broad <= 0.75 for broad, obj in rates)


def _candidate_edges(
    *,
    scene_base: pd.DataFrame,
    support_features: dict[str, np.ndarray],
    variant: dict[str, Any],
    control_role: str,
    rng: np.random.Generator,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    object_ids = sorted(scene_base["mv_object_id"].astype(str).unique().tolist())
    features = np.stack([support_features[oid] for oid in object_ids], axis=0)
    if control_role == "shuffled_support_control" and len(object_ids) > 1:
        features = features[rng.permutation(len(object_ids))]
    support_sim = features @ features.T
    by_object = {oid: scene_base[scene_base["mv_object_id"].astype(str) == oid] for oid in object_ids}
    max_gap = int(variant.get("max_temporal_gap", 10))
    tau_s = float(variant["support_threshold"])
    stats = {
        "skeleton_edge_count": 0,
        "SS_confirmed_edge_count": 0,
        "SS_rejected_by_support_count": 0,
        "SS_rejected_by_veto_count": 0,
        "SS_rejected_by_semantic_gate_count": 0,
        "support_only_candidate_count": 0,
    }
    local_by_object: dict[str, list[dict[str, Any]]] = defaultdict(list)
    all_candidates: list[dict[str, Any]] = []
    for i, oid_a in enumerate(object_ids[:-1]):
        for j in range(i + 1, len(object_ids)):
            oid_b = object_ids[j]
            group_a = by_object[oid_a]
            group_b = by_object[oid_b]
            sk_score = _skeleton_score(group_a, group_b, max_gap=max_gap)
            if sk_score <= 0.0:
                continue
            stats["skeleton_edge_count"] += 1
            s_score = float(support_sim[i, j])
            support_density = 0.5 * (
                float(group_a["phase5_support_count"].astype(float).mean())
                + float(group_b["phase5_support_count"].astype(float).mean())
            )
            support_ok = s_score >= tau_s
            if control_role == "density_control":
                support_ok = True
            if not support_ok:
                stats["SS_rejected_by_support_count"] += 1
                continue
            if bool(variant.get("semantic_gate", False)) and not _semantic_gate_ok(group_a, group_b):
                stats["SS_rejected_by_semantic_gate_count"] += 1
                continue
            conflict = _specific_conflict(group_a, group_b)
            broad = _broad_support_risk(group_a, variant) or _broad_support_risk(group_b, variant)
            if conflict or broad:
                stats["SS_rejected_by_veto_count"] += 1
                continue
            stats["SS_confirmed_edge_count"] += 1
            row = {
                "object_a": oid_a,
                "object_b": oid_b,
                "skeleton_score": sk_score,
                "support_affinity": s_score,
                "support_density_score": support_density,
                "combined_score": 0.55 * sk_score + 0.45 * s_score,
                "temporal_gap": _temporal_gap(group_a, group_b),
                "specific_conflict": conflict,
                "broad_support_veto": broad,
                "semantic_gate_pass": _semantic_gate_ok(group_a, group_b),
                "support_only_edge_flag": False,
            }
            all_candidates.append(row)
            local_by_object[oid_a].append(row)
    if control_role == "density_control":
        real_like = sorted(all_candidates, key=lambda r: (r["combined_score"], r["support_affinity"]), reverse=True)
        keep_by_scene = max(1, min(len(real_like), int(np.ceil(0.20 * max(len(real_like), 1)))))
        density_ranked = sorted(all_candidates, key=lambda r: (r["support_density_score"], r["skeleton_score"]), reverse=True)
        allowed = {
            tuple(sorted((str(r["object_a"]), str(r["object_b"]))))
            for r in density_ranked[:keep_by_scene]
        }
        for oid in list(local_by_object):
            local_by_object[oid] = [
                r for r in local_by_object[oid] if tuple(sorted((str(r["object_a"]), str(r["object_b"])))) in allowed
            ]
    edges: list[dict[str, Any]] = []
    topk = int(variant.get("topk_per_object", 1))
    for rows in local_by_object.values():
        if control_role == "density_control":
            rows.sort(key=lambda row: (float(row["support_density_score"]), float(row["skeleton_score"])), reverse=True)
        else:
            rows.sort(key=lambda row: (float(row["combined_score"]), float(row["support_affinity"])), reverse=True)
        edges.extend(rows[:topk] if topk > 0 else rows)
    sort_key = "support_density_score" if control_role == "density_control" else "combined_score"
    edges.sort(key=lambda row: (float(row[sort_key]), float(row["skeleton_score"])), reverse=True)
    return edges, stats


class _MaskGtCache:
    def __init__(self, phase2_summaries: dict[str, dict[str, Any]]) -> None:
        self.phase2_summaries = phase2_summaries
        self.labels: dict[tuple[str, int], np.ndarray] = {}
        self.gt: dict[tuple[str, int], np.ndarray] = {}

    def label(self, scene: str, frame: int) -> np.ndarray:
        key = (scene, int(frame))
        if key not in self.labels:
            self.labels[key] = _load_label(_label_path(str(self.phase2_summaries[scene]["mask_root"]), int(frame)))
        return self.labels[key]

    def gt_2d(self, scene: str, frame: int, shape: tuple[int, int]) -> np.ndarray:
        key = (scene, int(frame))
        if key not in self.gt:
            self.gt[key] = _load_gt_2d(scene, int(frame), shape)
        return self.gt[key]


def _object_gt_stats(scene: str, rows: pd.DataFrame, cache: _MaskGtCache) -> dict[str, Any]:
    inter: defaultdict[int, int] = defaultdict(int)
    gt_area: defaultdict[int, int] = defaultdict(int)
    mask_area = 0
    missing = 0
    for frame, group in rows.groupby("frame_id", sort=True):
        label = cache.label(scene, int(frame))
        gt = cache.gt_2d(scene, int(frame), label.shape)
        pixels = np.zeros(label.shape, dtype=bool)
        for row in group.to_dict("records"):
            mask = label == int(row["selected_mask_id"])
            if not np.any(mask):
                missing += 1
            pixels |= mask
        mask_area += int(np.count_nonzero(pixels))
        if np.any(gt > 0):
            ids, counts = np.unique(gt[gt > 0], return_counts=True)
            for gid, count in zip(ids, counts):
                gt_area[int(gid)] += int(count)
        both = pixels & (gt > 0)
        if np.any(both):
            ids, counts = np.unique(gt[both], return_counts=True)
            for gid, count in zip(ids, counts):
                inter[int(gid)] += int(count)
    best_gt = 0
    best_iou = 0.0
    for gid, value in inter.items():
        union = int(mask_area) + int(gt_area.get(gid, 0)) - int(value)
        iou = float(value) / float(union) if union > 0 else 0.0
        if iou > best_iou:
            best_iou = iou
            best_gt = int(gid)
    return {
        "mask_area": int(mask_area),
        "missing_mask_count": int(missing),
        "primary_gt_id": int(best_gt),
        "primary_gt_iou": float(best_iou),
        "inter": dict(inter),
        "gt_area": dict(gt_area),
    }


def _union_gt_stats(scene: str, rows_a: pd.DataFrame, rows_b: pd.DataFrame, cache: _MaskGtCache) -> dict[str, Any]:
    rows = pd.concat([rows_a, rows_b], ignore_index=True)
    return _object_gt_stats(scene, rows, cache)


def _materialize_variant(
    *,
    base: pd.DataFrame,
    support_features: dict[str, np.ndarray],
    variant: dict[str, Any],
    chunk_id: str,
    control_role: str,
    density_target_edges: dict[str, int] | None,
    phase2_summaries: dict[str, dict[str, Any]],
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    scene_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    edge_rows: list[dict[str, Any]] = []
    cluster_rows: list[dict[str, Any]] = []
    cache = _MaskGtCache(phase2_summaries)
    all_stats: dict[str, Any] = {
        "skeleton_edge_count": 0,
        "SS_confirmed_edge_count": 0,
        "SS_rejected_by_support_count": 0,
        "SS_rejected_by_veto_count": 0,
        "SS_rejected_by_semantic_gate_count": 0,
        "support_only_candidate_count": 0,
        "accepted_S_only_edge_count": 0,
        "accepted_edge_count": 0,
        "same_GT_connection_count_diagnostic": 0,
        "diff_GT_false_connection_count_diagnostic": 0,
        "gt_known_connection_count_diagnostic": 0,
        "union_minus_best_IoU_values": [],
    }
    rng = np.random.default_rng(7404)
    for scene, scene_base in base.groupby("scene_id", sort=True):
        scene = str(scene)
        object_ids = sorted(scene_base["mv_object_id"].astype(str).unique().tolist())
        by_object = {oid: scene_base[scene_base["mv_object_id"].astype(str) == oid] for oid in object_ids}
        uf = UnionFind(object_ids)
        edges, stats = _candidate_edges(
            scene_base=scene_base,
            support_features=support_features,
            variant=variant,
            control_role=control_role,
            rng=rng,
        )
        if control_role == "density_control" and density_target_edges is not None:
            target = int(density_target_edges.get(scene, 0))
            edges = edges[:target]
        for key, value in stats.items():
            all_stats[key] += int(value)
        for rank, edge in enumerate(edges):
            accepted = False
            reason = ""
            edge_diag: dict[str, Any] = {}
            if uf.find(str(edge["object_a"])) != uf.find(str(edge["object_b"])):
                uf.union(str(edge["object_a"]), str(edge["object_b"]))
                accepted = True
                all_stats["accepted_edge_count"] += 1
                a_stats = _object_gt_stats(scene, by_object[str(edge["object_a"])], cache)
                b_stats = _object_gt_stats(scene, by_object[str(edge["object_b"])], cache)
                u_stats = _union_gt_stats(scene, by_object[str(edge["object_a"])], by_object[str(edge["object_b"])], cache)
                a_gt = int(a_stats["primary_gt_id"])
                b_gt = int(b_stats["primary_gt_id"])
                same_gt = bool(a_gt > 0 and a_gt == b_gt)
                diff_gt = bool(a_gt > 0 and b_gt > 0 and a_gt != b_gt)
                if a_gt > 0 and b_gt > 0:
                    all_stats["gt_known_connection_count_diagnostic"] += 1
                    all_stats["same_GT_connection_count_diagnostic"] += int(same_gt)
                    all_stats["diff_GT_false_connection_count_diagnostic"] += int(diff_gt)
                union_minus_best = float(u_stats["primary_gt_iou"]) - max(
                    float(a_stats["primary_gt_iou"]),
                    float(b_stats["primary_gt_iou"]),
                )
                all_stats["union_minus_best_IoU_values"].append(union_minus_best)
                edge_diag = {
                    "object_a_primary_gt_id_diagnostic": a_gt,
                    "object_b_primary_gt_id_diagnostic": b_gt,
                    "union_primary_gt_id_diagnostic": int(u_stats["primary_gt_id"]),
                    "object_a_primary_gt_iou_diagnostic": float(a_stats["primary_gt_iou"]),
                    "object_b_primary_gt_iou_diagnostic": float(b_stats["primary_gt_iou"]),
                    "union_primary_gt_iou_diagnostic": float(u_stats["primary_gt_iou"]),
                    "same_GT_connection_diagnostic": same_gt,
                    "diff_GT_false_connection_diagnostic": diff_gt,
                    "union_minus_best_IoU_diagnostic": union_minus_best,
                }
            edge_rows.append(
                {
                    "schema_version": "stream4d_v103_r7_phase4_edge_row_v1",
                    "phase_id": PHASE_ID,
                    "variant_id": str(variant["variant_id"]),
                    "control_role": control_role,
                    "scene_id": scene,
                    "edge_rank": int(rank),
                    "object_a": str(edge["object_a"]),
                    "object_b": str(edge["object_b"]),
                    "skeleton_edge_source": "f2_replay_temporal_mask_view_neighborhood",
                    "has_mask_view_skeleton_edge": True,
                    "skeleton_score": float(edge["skeleton_score"]),
                    "temporal_gap": int(edge["temporal_gap"]),
                    "support_affinity": float(edge["support_affinity"]),
                    "support_density_score": float(edge["support_density_score"]),
                    "combined_score": float(edge["combined_score"]),
                    "accepted_union": bool(accepted),
                    "reject_reason": reason,
                    "specific_conflict": bool(edge["specific_conflict"]),
                    "broad_support_veto": bool(edge["broad_support_veto"]),
                    "semantic_gate_pass": bool(edge["semantic_gate_pass"]),
                    "support_only_edge_flag": False,
                    "uses_gt_for_prediction": False,
                    "uses_gt_for_diagnostic": bool(accepted),
                    "uses_future": False,
                    **edge_diag,
                }
            )
        components: dict[str, list[str]] = defaultdict(list)
        for oid in object_ids:
            components[uf.find(oid)].append(oid)
        for comp_idx, members in enumerate(sorted(components.values(), key=lambda v: (-len(v), v[0]))):
            comp_rows = scene_base[scene_base["mv_object_id"].astype(str).isin(members)].copy()
            object_id = f"{variant['variant_id']}:{control_role}:{scene}:{chunk_id}:merged_{comp_idx:05d}"
            frames = sorted(comp_rows["frame_id"].astype(int).unique().tolist())
            base_score = float(comp_rows.get("score", pd.Series([0.0])).max())
            cluster_rows.append(
                {
                    "schema_version": "stream4d_v103_r7_phase4_cluster_row_v1",
                    "phase_id": PHASE_ID,
                    "variant_id": str(variant["variant_id"]),
                    "control_role": control_role,
                    "scene_id": scene,
                    "mv_object_id": object_id,
                    "source_object_ids": json.dumps(members, sort_keys=True),
                    "source_object_count": int(len(members)),
                    "frame_count": int(len(frames)),
                    "object_score": base_score,
                    "base_object_score": base_score,
                    "score_policy": "max_f2_score",
                    "uses_gt_for_prediction": False,
                }
            )
            for _frame, group in comp_rows.groupby("frame_id", sort=True):
                best = sorted(
                    group.to_dict("records"),
                    key=lambda row: (
                        -float(row.get("score", 0.0)),
                        -int(row.get("phase5_support_count", 0)),
                        -int(bool(row.get("selected_mask_is_object_like", False))),
                        int(bool(row.get("selected_mask_is_broad", True))),
                        int(row.get("selected_mask_id", 0)),
                    ),
                )[0]
                scene_rows[scene].append(
                    {
                        "schema_version": "stream4d_v103_r7_phase4_frame_mask_row_v1",
                        "phase_id": PHASE_ID,
                        "variant_id": str(variant["variant_id"]),
                        "control_role": control_role,
                        "mv_object_id": object_id,
                        "object_id": object_id,
                        "scene_id": scene,
                        "chunk_id": str(best.get("chunk_id", chunk_id)),
                        "window_id": str(best.get("window_id", chunk_id)),
                        "frame_local_index": int(best["frame_local_index"]),
                        "frame_id": int(best["frame_id"]),
                        "selected_mask_id": int(best["selected_mask_id"]),
                        "mask_id_or_generated_id": int(best["mask_id_or_generated_id"]),
                        "object_score": base_score,
                        "score": base_score,
                        "support_count": int(best.get("phase5_support_count", 0) or 0),
                        "node_policy": "r7_skeleton_confirmed_support",
                        "emit_policy": "component_wta_by_f2_score_support",
                        "readout_mode": "r7_skeleton_confirmed_support",
                        "uses_gt_for_prediction": False,
                        "uses_future": False,
                    }
                )
    known = int(all_stats["gt_known_connection_count_diagnostic"])
    all_stats["same_GT_connection_rate_diagnostic"] = (
        float(all_stats["same_GT_connection_count_diagnostic"]) / max(known, 1)
    )
    all_stats["diff_GT_false_connection_rate_diagnostic"] = (
        float(all_stats["diff_GT_false_connection_count_diagnostic"]) / max(known, 1)
    )
    vals = [float(v) for v in all_stats.pop("union_minus_best_IoU_values")]
    all_stats["union_minus_best_IoU_mean"] = float(np.mean(vals)) if vals else 0.0
    return scene_rows, edge_rows, cluster_rows, all_stats


def _density_targets(edge_rows: list[dict[str, Any]]) -> dict[str, int]:
    out: defaultdict[str, int] = defaultdict(int)
    for row in edge_rows:
        if bool(row.get("accepted_union", False)):
            out[str(row["scene_id"])] += 1
    return dict(out)


def build(args: argparse.Namespace) -> dict[str, Any]:
    t0 = time.time()
    out = _project(args.output_root)
    if out.exists() and any(out.iterdir()) and not args.force:
        raise SystemExit(f"output root already exists and is non-empty; pass --force: {_rel(out)}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "last_command.txt").write_text(" ".join([sys.executable, str(Path(__file__)), *sys.argv[1:]]) + "\n", encoding="utf-8")

    phase1_root = _project(args.r7_phase1_root)
    phase1_summary = _read_json(phase1_root / "summary.json")
    f2_root = _project(args.f2_root)
    support_root = _project(args.support_feature_root)
    phase2_summaries = {
        "scene0011_00": _read_json(_project(args.scene0011_phase2_root) / "summary.json"),
        "scene0050_00": _read_json(_project(args.scene0050_phase2_root) / "summary.json"),
    }
    support_payloads = {scene: _load_phase5_scene(support_root, scene) for scene in phase2_summaries}
    base = _adapt_f2_rows(
        f2_root=f2_root,
        phase2_summaries=phase2_summaries,
        phase5_payloads=support_payloads,
        dataset_split=str(args.dataset_split),
        chunk_id=str(args.chunk_id),
    )
    _, support_features = _object_tables(base, support_payloads)

    forbidden_tokens = ["da3", "3dgs", "gaussian", "phase9n", "phase9b", "phase9c", "phase9d", "da3_pair"]
    inspected_paths = [phase1_root, f2_root, support_root, _project(args.scene0011_phase2_root), _project(args.scene0050_phase2_root)]
    forbidden_hits = [
        _rel(path)
        for path in inspected_paths
        if any(token in _rel(path).lower() for token in forbidden_tokens)
    ]

    artifact_rows = [
        _artifact_row("r7_phase1_summary", phase1_root / "summary.json"),
        _artifact_row("f2_replay_rows", f2_root / "mv_object_frame_mask_rows.parquet", note="F2 replay object-frame rows used to infer temporal mask-view skeleton candidate edges."),
        _artifact_row("support_feature_root", support_root, note="R6F2 D4RT-only support-conditioned semantic phase5-like feature root"),
        _artifact_row("scene0011_phase2_summary", _project(args.scene0011_phase2_root) / "summary.json"),
        _artifact_row("scene0050_phase2_summary", _project(args.scene0050_phase2_root) / "summary.json"),
        _artifact_row("last_command", out / "last_command.txt", required=False),
    ]

    metric_rows: list[dict[str, Any]] = []
    control_rows: list[dict[str, Any]] = []
    variant_rows: list[dict[str, Any]] = []
    all_edge_rows: list[dict[str, Any]] = []
    all_cluster_rows: list[dict[str, Any]] = []
    all_selected_rows: list[dict[str, Any]] = []
    all_window_rows: list[dict[str, Any]] = []
    best_real: dict[str, Any] | None = None

    for variant in VARIANTS:
        real_scene_rows, real_edges, real_clusters, real_stats = _materialize_variant(
            base=base,
            support_features=support_features,
            variant=variant,
            chunk_id=str(args.chunk_id),
            control_role="real",
            density_target_edges=None,
            phase2_summaries=phase2_summaries,
        )
        density_target = _density_targets(real_edges)
        for control_role, scene_rows, edge_rows, cluster_rows, stats in [
            ("real", real_scene_rows, real_edges, real_clusters, real_stats),
            (
                "shuffled_support_control",
                *_materialize_variant(
                    base=base,
                    support_features=support_features,
                    variant=variant,
                    chunk_id=str(args.chunk_id),
                    control_role="shuffled_support_control",
                    density_target_edges=None,
                    phase2_summaries=phase2_summaries,
                ),
            ),
            (
                "density_control",
                *_materialize_variant(
                    base=base,
                    support_features=support_features,
                    variant=variant,
                    chunk_id=str(args.chunk_id),
                    control_role="density_control",
                    density_target_edges=density_target,
                    phase2_summaries=phase2_summaries,
                ),
            ),
        ]:
            eval_variant_id = str(variant["variant_id"]) if control_role == "real" else f"{variant['variant_id']}__{control_role}"
            window_rows, aggregate, selected_rows, pixel_collision_count, missing_count, _frame_count = _evaluate_variant(
                variant_id=eval_variant_id,
                scene_rows=scene_rows,
                phase2_summaries=phase2_summaries,
                min_pred_pixels=int(args.min_pred_pixels),
                min_gt_pixels=int(args.min_gt_pixels),
                use_cupy_iou=not bool(args.disable_cupy_iou),
                cupy_device_id=int(args.cupy_device_id),
            )
            row = {
                "schema_version": "stream4d_v103_r7_phase4_metric_row_v1",
                "phase_id": PHASE_ID,
                "variant_id": eval_variant_id,
                "base_variant_id": str(variant["variant_id"]),
                "variant_family": "R7SS_skeleton_confirmed_support",
                "control_role": control_role,
                "support_threshold": float(variant["support_threshold"]),
                "max_temporal_gap": int(variant.get("max_temporal_gap", 10)),
                "skeleton_edge_source": "f2_replay_temporal_mask_view_neighborhood",
                "skeleton_edge_count": int(stats["skeleton_edge_count"]),
                "SS_confirmed_edge_count": int(stats["SS_confirmed_edge_count"]),
                "SS_rejected_by_support_count": int(stats["SS_rejected_by_support_count"]),
                "SS_rejected_by_veto_count": int(stats["SS_rejected_by_veto_count"]),
                "SS_rejected_by_semantic_gate_count": int(stats["SS_rejected_by_semantic_gate_count"]),
                "accepted_S_only_edge_count": int(stats["accepted_S_only_edge_count"]),
                "accepted_edge_count": int(stats["accepted_edge_count"]),
                "same_GT_connection_rate_diagnostic": float(stats["same_GT_connection_rate_diagnostic"]),
                "diff_GT_false_connection_rate_diagnostic": float(stats["diff_GT_false_connection_rate_diagnostic"]),
                "gt_known_connection_count_diagnostic": int(stats["gt_known_connection_count_diagnostic"]),
                "union_minus_best_IoU_mean": float(stats["union_minus_best_IoU_mean"]),
                "GT_fragment_count_mean": float(np.mean([float(w.get("gt_fragment_count_mean", 0.0)) for w in window_rows])) if window_rows else 0.0,
                "GT_fragment_count_ge2_rate": float(np.mean([float(w.get("gt_fragment_count_ge2_rate", 0.0)) for w in window_rows])) if window_rows else 0.0,
                "same_frame_collision_count": int(aggregate.get("same_frame_collision_count", 0)),
                "pixel_collision_rate": float(aggregate.get("pixel_collision_rate", 0.0)),
                "missing_mask_raster_count": int(missing_count),
                "MV_AP_window": float(aggregate.get("MV_AP_window", 0.0)),
                "MV_AP50_window": float(aggregate.get("MV_AP50_window", 0.0)),
                "MV_AP25_window": float(aggregate.get("MV_AP25_window", 0.0)),
                "ScoreFreeMatch50_window": float(aggregate.get("ScoreFreeMatch50_window", 0.0)),
                "uses_gt_for_prediction": False,
                "uses_gt_for_eval": True,
                "uses_gt_for_diagnostic": True,
                "uses_future": False,
            }
            metric_rows.append(row)
            if control_role != "real":
                control_rows.append(row)
            elif best_real is None or float(row["MV_AP_window"]) > float(best_real["MV_AP_window"]):
                best_real = row
            variant_rows.append(
                {
                    "schema_version": "stream4d_v103_r7_phase4_variant_row_v1",
                    "phase_id": PHASE_ID,
                    "variant_id": eval_variant_id,
                    "control_role": control_role,
                    "definition": json.dumps(_jsonable(variant), sort_keys=True),
                    "uses_gt_for_prediction": False,
                    "uses_future": False,
                }
            )
            for edge in edge_rows:
                edge["variant_id"] = eval_variant_id
            for cluster in cluster_rows:
                cluster["variant_id"] = eval_variant_id
            all_edge_rows.extend(edge_rows)
            all_cluster_rows.extend(cluster_rows)
            all_selected_rows.extend(selected_rows)
            all_window_rows.extend(window_rows)

    by_id = {str(row["variant_id"]): row for row in metric_rows}
    replay_mv = _num(phase1_summary.get("replay_MV_AP_window"), 0.1147308155349091)
    replay_ap50 = _num(phase1_summary.get("replay_MV_AP50_window"), 0.2871561795903901)
    baseline_window = next((r for r in metric_rows if str(r["base_variant_id"]) == "R7SS1_skeleton_support_confirm_tauS070" and str(r["control_role"]) == "shuffled_support_control"), {})
    diff_gt_reference_available = "D9_reference_diff_GT_false_connection_rate_diagnostic" in phase1_summary
    r6_reference_diff_gt = _num(phase1_summary.get("D9_reference_diff_GT_false_connection_rate_diagnostic"), 0.0)
    for row in metric_rows:
        if row["control_role"] != "real":
            continue
        shuf = by_id.get(f"{row['base_variant_id']}__shuffled_support_control", {})
        dens = by_id.get(f"{row['base_variant_id']}__density_control", {})
        best_control_mv = max(_num(shuf.get("MV_AP_window"), 0.0), _num(dens.get("MV_AP_window"), 0.0))
        best_control_ap50 = max(_num(shuf.get("MV_AP50_window"), 0.0), _num(dens.get("MV_AP50_window"), 0.0))
        row["real_minus_shuffled_MV_AP_window"] = float(row["MV_AP_window"]) - _num(shuf.get("MV_AP_window"), 0.0)
        row["real_minus_density_control_MV_AP_window"] = float(row["MV_AP_window"]) - _num(dens.get("MV_AP_window"), 0.0)
        row["real_minus_best_control_MV_AP_window"] = float(row["MV_AP_window"]) - best_control_mv
        row["real_minus_best_control_MV_AP50_window"] = float(row["MV_AP50_window"]) - best_control_ap50
        row["real_minus_replay_MV_AP_window"] = float(row["MV_AP_window"]) - replay_mv
        row["real_minus_replay_MV_AP50_window"] = float(row["MV_AP50_window"]) - replay_ap50

    best_real = best_real or {}
    best_real_full = by_id.get(str(best_real.get("variant_id", "")), best_real)
    best_real_minus_control = _num(best_real_full.get("real_minus_best_control_MV_AP_window"), 0.0)
    best_real_minus_control_ap50 = _num(best_real_full.get("real_minus_best_control_MV_AP50_window"), 0.0)
    fragmentation_baseline = _num(baseline_window.get("GT_fragment_count_ge2_rate"), 0.0)
    same_gt_baseline = _num(baseline_window.get("same_GT_connection_rate_diagnostic"), 0.0)
    fragmentation_improved = _num(best_real_full.get("GT_fragment_count_ge2_rate"), 0.0) < fragmentation_baseline
    same_gt_increased = _num(best_real_full.get("same_GT_connection_rate_diagnostic"), 0.0) > same_gt_baseline
    subset_gate_pass = bool(
        best_real_full
        and not forbidden_hits
        and _num(best_real_full.get("MV_AP_window"), 0.0) >= replay_mv + 0.005
        and _num(best_real_full.get("MV_AP50_window"), 0.0) >= replay_ap50 + 0.010
        and best_real_minus_control >= 0.003
        and diff_gt_reference_available
        and _num(best_real_full.get("diff_GT_false_connection_rate_diagnostic"), 1.0) <= r6_reference_diff_gt
        and (fragmentation_improved or same_gt_increased)
        and int(best_real_full.get("accepted_S_only_edge_count", 1)) == 0
        and int(best_real_full.get("same_frame_collision_count", 1)) == 0
        and int(best_real_full.get("missing_mask_raster_count", 1)) == 0
        and float(best_real_full.get("pixel_collision_rate", 1.0)) == 0.0
    )
    diagnostic_net_improved = bool(
        (fragmentation_improved or same_gt_increased)
        and _num(best_real_full.get("same_GT_connection_rate_diagnostic"), 0.0) >= 0.5
        and _num(best_real_full.get("diff_GT_false_connection_rate_diagnostic"), 1.0) <= 0.25
        and _num(best_real_full.get("union_minus_best_IoU_mean"), -1.0) >= 0.0
    )
    history_candidate = bool(
        best_real_full
        and not subset_gate_pass
        and diagnostic_net_improved
        and int(best_real_full.get("accepted_S_only_edge_count", 1)) == 0
        and int(best_real_full.get("same_frame_collision_count", 1)) == 0
        and int(best_real_full.get("missing_mask_raster_count", 1)) == 0
        and float(best_real_full.get("pixel_collision_rate", 1.0)) == 0.0
    )

    gate_rows = [
        _gate_row("phase1_d4rt_only", not bool(phase1_summary.get("DA3_USED", True)), phase1_summary.get("DA3_USED", ""), False, "Repair R7-1 D4RT-only boundary."),
        _gate_row("no_forbidden_artifact_path_tokens", not forbidden_hits, forbidden_hits, [], "Remove DA3/3DGS/Gaussian/phase9 provider artifacts from R7-4 inputs."),
        _gate_row("accepted_S_only_edge_count_zero", all(int(r.get("accepted_S_only_edge_count", 0)) == 0 for r in metric_rows), 0, 0, "Disallow support-only union."),
        _gate_row("best_MV_AP_window_ge_replay_plus_0p005", _num(best_real_full.get("MV_AP_window"), 0.0) >= replay_mv + 0.005, best_real_full.get("MV_AP_window", ""), replay_mv + 0.005, "Try R7-5 propagation if false."),
        _gate_row("best_MV_AP50_window_ge_replay_plus_0p010", _num(best_real_full.get("MV_AP50_window"), 0.0) >= replay_ap50 + 0.010, best_real_full.get("MV_AP50_window", ""), replay_ap50 + 0.010, "Try R7-5 propagation if false."),
        _gate_row("real_minus_best_control_MV_AP_window_ge_0p003", best_real_minus_control >= 0.003, best_real_minus_control, 0.003, "Inspect shuffled/density controls; do not promote if control explains gain."),
        _gate_row("diff_GT_reference_available", diff_gt_reference_available, sorted(phase1_summary.keys()), "D9_reference_diff_GT_false_connection_rate_diagnostic", "Add R6/R7-1 edge-level diff-GT reference before formal R7-4 promotion."),
        _gate_row("diff_GT_false_connection_rate_le_reference", diff_gt_reference_available and _num(best_real_full.get("diff_GT_false_connection_rate_diagnostic"), 1.0) <= r6_reference_diff_gt, best_real_full.get("diff_GT_false_connection_rate_diagnostic", ""), f"<= {r6_reference_diff_gt} with reference_available={diff_gt_reference_available}", "Tighten skeleton/veto if false."),
        _gate_row("fragmentation_or_same_gt_improves", fragmentation_improved or same_gt_increased, {"fragmentation_improved": fragmentation_improved, "same_gt_increased": same_gt_increased}, "true", "Check stricter diagnostic_net_improved gate before retaining as history candidate."),
        _gate_row("diagnostic_net_improved_for_history_candidate", diagnostic_net_improved, {"same_GT_connection_rate": best_real_full.get("same_GT_connection_rate_diagnostic", ""), "diff_GT_false_connection_rate": best_real_full.get("diff_GT_false_connection_rate_diagnostic", ""), "union_minus_best_IoU_mean": best_real_full.get("union_minus_best_IoU_mean", "")}, "same_GT>=0.5 diff_GT<=0.25 union_minus_best>=0", "Do not retain as history candidate if false; proceed to R7-5."),
        _gate_row("collision_missing_safe", int(best_real_full.get("same_frame_collision_count", 1)) == 0 and int(best_real_full.get("missing_mask_raster_count", 1)) == 0 and float(best_real_full.get("pixel_collision_rate", 1.0)) == 0.0, best_real_full, "collision=0 missing=0 pixel_collision_rate=0", "Repair materialization if false."),
    ]

    failure_rows: list[dict[str, Any]] = []
    if forbidden_hits:
        failure_rows.append(
            _failure_row(
                "R7_DA3_ARTIFACT_LEAKAGE",
                {"forbidden_hits": forbidden_hits},
                "Terminate contaminated variant and rerun with D4RT-only inputs.",
            )
        )
    if not subset_gate_pass:
        failure_rows.append(
            _failure_row(
                "NO_GO_R7_4_SKELETON_CONFIRMED_SUPPORT_SUBSET_GATE",
                {
                    "best_real": best_real_full,
                    "replay_MV_AP_window": replay_mv,
                    "replay_MV_AP50_window": replay_ap50,
                    "history_candidate": history_candidate,
                },
                "Proceed to R7-5 anchor-seeded support propagation; retain R7-4 only as history candidate if 3D inconsistency improves.",
            )
        )
    else:
        failure_rows.append(_failure_row("NONE", "R7-4 subset gate passed.", "Freeze config before full-dev.", severity="info"))

    _write_csv(out / "artifact_rows.csv", artifact_rows)
    _write_csv(out / "variant_rows.csv", variant_rows)
    _write_csv(out / "metric_rows.csv", metric_rows)
    _write_csv(out / "control_rows.csv", control_rows)
    _write_csv(out / "gate_rows.csv", gate_rows)
    _write_csv(out / "failure_rows.csv", failure_rows)
    _write_csv(out / "edge_rows.csv", all_edge_rows)
    _write_csv(out / "cluster_rows.csv", all_cluster_rows)
    _write_csv(out / "selected_rows.csv", all_selected_rows)
    _write_csv(out / "window_rows.csv", all_window_rows)

    summary = {
        "schema_version": "stream4d_v103_r7_phase4_summary_v1",
        "phase": "R7-4",
        "phase_id": PHASE_ID,
        "phase_pass": subset_gate_pass,
        "history_candidate": history_candidate,
        "decision": "PASS_R7_4_SUBSET_CANDIDATE" if subset_gate_pass else "NO_GO_R7_4_SKELETON_CONFIRMED_SUPPORT",
        "runtime_sec": time.time() - t0,
        "variant_count": len(VARIANTS),
        "metric_row_count": len(metric_rows),
        "edge_row_count": len(all_edge_rows),
        "skeleton_edge_source": "f2_replay_temporal_mask_view_neighborhood",
        "best_real_variant_id": best_real_full.get("variant_id", ""),
        "best_real_MV_AP_window": best_real_full.get("MV_AP_window", ""),
        "best_real_MV_AP50_window": best_real_full.get("MV_AP50_window", ""),
        "best_real_minus_replay_MV_AP_window": _num(best_real_full.get("MV_AP_window"), 0.0) - replay_mv if best_real_full else "",
        "best_real_minus_replay_MV_AP50_window": _num(best_real_full.get("MV_AP50_window"), 0.0) - replay_ap50 if best_real_full else "",
        "best_real_minus_best_control_MV_AP_window": best_real_minus_control,
        "best_real_minus_best_control_MV_AP50_window": best_real_minus_control_ap50,
        "accepted_S_only_edge_count": int(best_real_full.get("accepted_S_only_edge_count", 0)) if best_real_full else "",
        "same_GT_connection_rate_diagnostic": best_real_full.get("same_GT_connection_rate_diagnostic", ""),
        "diff_GT_false_connection_rate_diagnostic": best_real_full.get("diff_GT_false_connection_rate_diagnostic", ""),
        "GT_fragment_count_ge2_rate": best_real_full.get("GT_fragment_count_ge2_rate", ""),
        "union_minus_best_IoU_mean": best_real_full.get("union_minus_best_IoU_mean", ""),
        "same_frame_collision_count": int(best_real_full.get("same_frame_collision_count", 0)) if best_real_full else "",
        "pixel_collision_rate": float(best_real_full.get("pixel_collision_rate", 0.0)) if best_real_full else "",
        "missing_mask_raster_count": int(best_real_full.get("missing_mask_raster_count", 0)) if best_real_full else "",
        "DA3_USED": False,
        "DA3_ROWS_LOADED": False,
        "GS_USED": False,
        "forbidden_token_hits": forbidden_hits,
        "uses_gt_for_prediction": False,
        "uses_gt_for_eval": True,
        "uses_gt_for_diagnostic": True,
        "uses_future": False,
        "outputs": {
            "summary": _rel(out / "summary.json"),
            "gate_rows": _rel(out / "gate_rows.csv"),
            "failure_rows": _rel(out / "failure_rows.csv"),
            "variant_rows": _rel(out / "variant_rows.csv"),
            "metric_rows": _rel(out / "metric_rows.csv"),
            "control_rows": _rel(out / "control_rows.csv"),
            "artifact_rows": _rel(out / "artifact_rows.csv"),
            "edge_rows": _rel(out / "edge_rows.csv"),
            "cluster_rows": _rel(out / "cluster_rows.csv"),
            "selected_rows": _rel(out / "selected_rows.csv"),
            "window_rows": _rel(out / "window_rows.csv"),
        },
        "truthfulness_note": "R7-4 infers mask-view temporal skeleton candidates from F2 replay object-frame rows, confirms only with D4RT-only S_support compatibility, and uses GT only for evaluator/edge diagnostics.",
    }
    _write_json(out / "summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build v103 R7-4 D4RT-only skeleton-confirmed support variants.")
    parser.add_argument("--output-root", default=str(DEFAULT_OUT))
    parser.add_argument("--r7-phase1-root", default=str(DEFAULT_R7_PHASE1_ROOT))
    parser.add_argument("--f2-root", default=str(DEFAULT_F2_ROOT))
    parser.add_argument("--support-feature-root", default=str(DEFAULT_SUPPORT_FEATURE))
    parser.add_argument("--scene0011-phase2-root", default=str(DEFAULT_SCENE0011_PHASE2))
    parser.add_argument("--scene0050-phase2-root", default=str(DEFAULT_SCENE0050_PHASE2))
    parser.add_argument("--dataset-split", default="dev")
    parser.add_argument("--chunk-id", default="c0001")
    parser.add_argument("--min-pred-pixels", type=int, default=100)
    parser.add_argument("--min-gt-pixels", type=int, default=100)
    parser.add_argument("--disable-cupy-iou", action="store_true")
    parser.add_argument("--cupy-device-id", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    summary = build(parse_args())
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
