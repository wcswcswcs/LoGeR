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

import numpy as np
import pandas as pd


STREAM3D_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = STREAM3D_ROOT.parent
if str(STREAM3D_ROOT) not in sys.path:
    sys.path.insert(0, str(STREAM3D_ROOT))

from tools.build_v103_phase6_mask_clustering_local_object_birth import _evaluate_variant  # noqa: E402
from tools.build_v103_phase6d_f2_skeleton_affinity_merge import (  # noqa: E402
    UnionFind,
    _adapt_f2_rows,
    _broad_support_risk,
    _load_phase5_scene,
    _object_tables,
    _specific_conflict,
)
from tools.build_v103_r7_phase4_skeleton_confirmed_support import (  # noqa: E402
    DEFAULT_F2_ROOT,
    DEFAULT_R7_PHASE1_ROOT,
    DEFAULT_SCENE0011_PHASE2,
    DEFAULT_SCENE0050_PHASE2,
    _MaskGtCache,
    _object_gt_stats,
    _semantic_gate_ok,
    _temporal_gap,
    _union_gt_stats,
)


AUDIT_ROOT = STREAM3D_ROOT / "outputs" / "audit"
PHASE_ID = "v103_r7_phase5_anchor_seeded_support_propagation"
DEFAULT_OUT = AUDIT_ROOT / "v103_r7_phase5_anchor_seeded_support_propagation"
DEFAULT_PHASE5_ROOT = AUDIT_ROOT / "v103_supp_r6_phase2_support_conditioned_local_ap/phase5_like_features"
DEFAULT_ANCHOR_FEATURE = DEFAULT_PHASE5_ROOT / "R6F0_anchor_only_replay"
DEFAULT_SUPPORT_FEATURE = DEFAULT_PHASE5_ROOT / "R6F2_support010_specificity_semantic"


VARIANTS = [
    {
        "variant_id": "R7SP1_onehop_anchor_seeded_support_tauS070",
        "anchor_threshold": 0.90,
        "support_threshold": 0.70,
        "max_temporal_gap": 10,
        "topk_per_seed": 1,
        "broad_support_veto": True,
        "hop_mode": "onehop",
    },
    {
        "variant_id": "R7SP2_onehop_anchor_seeded_support_tauS060",
        "anchor_threshold": 0.90,
        "support_threshold": 0.60,
        "max_temporal_gap": 10,
        "topk_per_seed": 1,
        "broad_support_veto": True,
        "hop_mode": "onehop",
    },
    {
        "variant_id": "R7SP3_onehop_anchor_seeded_support_semantic_gate",
        "anchor_threshold": 0.90,
        "support_threshold": 0.60,
        "max_temporal_gap": 10,
        "topk_per_seed": 1,
        "broad_support_veto": True,
        "semantic_gate": True,
        "hop_mode": "onehop",
    },
    {
        "variant_id": "R7SP4_onehop_anchor_seeded_support_temporal_neighbor_only",
        "anchor_threshold": 0.90,
        "support_threshold": 0.60,
        "max_temporal_gap": 5,
        "topk_per_seed": 1,
        "broad_support_veto": True,
        "temporal_neighbor_only": True,
        "hop_mode": "onehop",
    },
    {
        "variant_id": "R7SP5_twohop_diagnostic_strict_tauS080",
        "anchor_threshold": 0.92,
        "support_threshold": 0.80,
        "max_temporal_gap": 5,
        "topk_per_seed": 1,
        "broad_support_veto": True,
        "semantic_gate": True,
        "hop_mode": "twohop_diagnostic",
        "eligible_for_full_dev": False,
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


def _sha256(path: Path) -> str:
    if not path.exists() or not path.is_file():
        return ""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _artifact_row(role: str, path: Path, *, required: bool = True, note: str = "") -> dict[str, Any]:
    exists = path.exists()
    row_count: int | str = ""
    if exists and path.is_file() and path.suffix.lower() in {".csv", ".parquet"}:
        try:
            row_count = max(sum(1 for _ in path.open("r", encoding="utf-8")) - 1, 0) if path.suffix.lower() == ".csv" else int(pd.read_parquet(path).shape[0])
        except Exception:
            row_count = ""
    return {
        "schema_version": "stream4d_v103_r7_phase5_artifact_row_v1",
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
        "schema_version": "stream4d_v103_r7_phase5_gate_row_v1",
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
        "schema_version": "stream4d_v103_r7_phase5_failure_row_v1",
        "phase_id": PHASE_ID,
        "failure_id": failure_id,
        "severity": severity,
        "detail": json.dumps(_jsonable(detail), sort_keys=True) if isinstance(detail, (dict, list, tuple)) else detail,
        "repair_direction": repair,
        "uses_gt_for_prediction": False,
        "uses_future": False,
    }


def _semantic_bucket(group: pd.DataFrame) -> str:
    broad = float(group["selected_mask_is_broad"].astype(bool).mean()) >= 0.5
    obj = float(group["selected_mask_is_object_like"].astype(bool).mean()) >= 0.5
    return f"object_like={int(obj)}|broad={int(broad)}"


def _blocked_pair(group_a: pd.DataFrame, group_b: pd.DataFrame, variant: dict[str, Any]) -> tuple[bool, str]:
    if _specific_conflict(group_a, group_b):
        return True, "specific_same_frame_conflict"
    if _broad_support_risk(group_a, variant) or _broad_support_risk(group_b, variant):
        return True, "broad_support_risk_veto"
    if bool(variant.get("semantic_gate", False)) and not _semantic_gate_ok(group_a, group_b):
        return True, "semantic_gate_reject"
    return False, ""


def _candidate_edges(
    *,
    scene_base: pd.DataFrame,
    anchor_features: dict[str, np.ndarray],
    support_features: dict[str, np.ndarray],
    variant: dict[str, Any],
    control_role: str,
    rng: np.random.Generator,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    object_ids = sorted(scene_base["mv_object_id"].astype(str).unique().tolist())
    anchor_matrix = np.stack([anchor_features[oid] for oid in object_ids], axis=0)
    support_matrix = np.stack([support_features[oid] for oid in object_ids], axis=0)
    if control_role == "shuffled_support_control" and len(object_ids) > 1:
        support_matrix = support_matrix[rng.permutation(len(object_ids))]
    anchor_sim = anchor_matrix @ anchor_matrix.T
    support_sim = support_matrix @ support_matrix.T
    by_object = {oid: scene_base[scene_base["mv_object_id"].astype(str) == oid] for oid in object_ids}
    tau_a = float(variant["anchor_threshold"])
    tau_s = float(variant["support_threshold"])
    max_gap = int(variant.get("max_temporal_gap", 10))
    stats = {
        "anchor_seed_edge_count": 0,
        "propagated_edge_count": 0,
        "onehop_edge_count": 0,
        "twohop_edge_count": 0,
        "propagated_edge_veto_reject_count": 0,
        "propagated_edge_semantic_reject_count": 0,
        "support_only_candidate_count": 0,
    }
    anchor_seeds: list[tuple[int, int, float]] = []
    for i, oid_a in enumerate(object_ids[:-1]):
        for j in range(i + 1, len(object_ids)):
            oid_b = object_ids[j]
            score = float(anchor_sim[i, j])
            if score < tau_a:
                continue
            blocked, reason = _blocked_pair(by_object[oid_a], by_object[oid_b], variant)
            if blocked:
                if reason == "semantic_gate_reject":
                    stats["propagated_edge_semantic_reject_count"] += 1
                else:
                    stats["propagated_edge_veto_reject_count"] += 1
                continue
            anchor_seeds.append((i, j, score))
            stats["anchor_seed_edge_count"] += 1
    best_by_pair: dict[tuple[str, str], dict[str, Any]] = {}
    for i, c, a_score in anchor_seeds:
        for root_idx, hub_idx in [(i, c), (c, i)]:
            root = object_ids[root_idx]
            hub = object_ids[hub_idx]
            local: list[dict[str, Any]] = []
            for b_idx, target in enumerate(object_ids):
                if b_idx in {root_idx, hub_idx}:
                    continue
                if _temporal_gap(by_object[hub], by_object[target]) > max_gap:
                    continue
                support_score = float(support_sim[hub_idx, b_idx])
                density_score = 0.5 * (
                    float(by_object[hub]["phase5_support_count"].astype(float).mean())
                    + float(by_object[target]["phase5_support_count"].astype(float).mean())
                )
                support_ok = support_score >= tau_s or control_role == "density_control"
                if not support_ok:
                    stats["support_only_candidate_count"] += int(support_score >= tau_s)
                    continue
                blocked, reason = _blocked_pair(by_object[hub], by_object[target], variant)
                if blocked:
                    if reason == "semantic_gate_reject":
                        stats["propagated_edge_semantic_reject_count"] += 1
                    else:
                        stats["propagated_edge_veto_reject_count"] += 1
                    continue
                edge_root = root
                edge_target = target
                hop_mode = str(variant.get("hop_mode", "onehop"))
                if hop_mode == "twohop_diagnostic":
                    second_best = None
                    for d_idx, second in enumerate(object_ids):
                        if d_idx in {root_idx, hub_idx, b_idx}:
                            continue
                        if _temporal_gap(by_object[target], by_object[second]) > max_gap:
                            continue
                        s2 = float(support_sim[b_idx, d_idx])
                        if s2 < tau_s:
                            continue
                        blocked2, reason2 = _blocked_pair(by_object[target], by_object[second], variant)
                        if blocked2:
                            if reason2 == "semantic_gate_reject":
                                stats["propagated_edge_semantic_reject_count"] += 1
                            else:
                                stats["propagated_edge_veto_reject_count"] += 1
                            continue
                        score2 = 0.45 * a_score + 0.30 * support_score + 0.25 * s2
                        if second_best is None or score2 > float(second_best["combined_score"]):
                            second_best = {
                                "object_a": edge_root,
                                "object_b": second,
                                "anchor_seed_object_a": root,
                                "anchor_seed_object_c": hub,
                                "support_hub_object": target,
                                "anchor_affinity": a_score,
                                "support_affinity": support_score,
                                "support_affinity_second_hop": s2,
                                "support_density_score": density_score,
                                "combined_score": score2,
                                "hop_type": "twohop_diagnostic",
                            }
                    if second_best is None:
                        continue
                    local.append(second_best)
                    continue
                local.append(
                    {
                        "object_a": edge_root,
                        "object_b": edge_target,
                        "anchor_seed_object_a": root,
                        "anchor_seed_object_c": hub,
                        "support_hub_object": hub,
                        "anchor_affinity": a_score,
                        "support_affinity": support_score,
                        "support_affinity_second_hop": 0.0,
                        "support_density_score": density_score,
                        "combined_score": 0.50 * a_score + 0.50 * support_score,
                        "hop_type": "onehop",
                    }
                )
            sort_key = "support_density_score" if control_role == "density_control" else "combined_score"
            local.sort(key=lambda row: (float(row[sort_key]), float(row["anchor_affinity"])), reverse=True)
            for row in local[: int(variant.get("topk_per_seed", 1))]:
                pair = tuple(sorted((str(row["object_a"]), str(row["object_b"]))))
                old = best_by_pair.get(pair)
                if old is None or float(row["combined_score"]) > float(old["combined_score"]):
                    best_by_pair[pair] = row
    edges = sorted(best_by_pair.values(), key=lambda row: (float(row["combined_score"]), float(row["anchor_affinity"])), reverse=True)
    stats["propagated_edge_count"] = len(edges)
    stats["onehop_edge_count"] = sum(1 for row in edges if row["hop_type"] == "onehop")
    stats["twohop_edge_count"] = sum(1 for row in edges if row["hop_type"] == "twohop_diagnostic")
    return edges, stats


def _materialize_variant(
    *,
    base: pd.DataFrame,
    anchor_features: dict[str, np.ndarray],
    support_features: dict[str, np.ndarray],
    variant: dict[str, Any],
    chunk_id: str,
    control_role: str,
    phase2_summaries: dict[str, dict[str, Any]],
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    scene_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    edge_rows: list[dict[str, Any]] = []
    cluster_rows: list[dict[str, Any]] = []
    cache = _MaskGtCache(phase2_summaries)
    stats: dict[str, Any] = {
        "anchor_seed_edge_count": 0,
        "propagated_edge_count": 0,
        "onehop_edge_count": 0,
        "twohop_edge_count": 0,
        "propagated_edge_veto_reject_count": 0,
        "propagated_edge_semantic_reject_count": 0,
        "support_only_candidate_count": 0,
        "accepted_S_only_edge_count": 0,
        "accepted_edge_count": 0,
        "accepted_diff_gt_edge_count_diagnostic": 0,
        "same_GT_connection_count_diagnostic": 0,
        "gt_known_connection_count_diagnostic": 0,
        "same_semantic_diff_gt_count_diagnostic": 0,
        "same_semantic_known_count_diagnostic": 0,
        "union_minus_best_IoU_values": [],
        "best_pred_IoU_values": [],
        "union_pred_IoU_values": [],
    }
    rng = np.random.default_rng(7505)
    for scene, scene_base in base.groupby("scene_id", sort=True):
        scene = str(scene)
        object_ids = sorted(scene_base["mv_object_id"].astype(str).unique().tolist())
        by_object = {oid: scene_base[scene_base["mv_object_id"].astype(str) == oid] for oid in object_ids}
        uf = UnionFind(object_ids)
        edges, local_stats = _candidate_edges(
            scene_base=scene_base,
            anchor_features=anchor_features,
            support_features=support_features,
            variant=variant,
            control_role=control_role,
            rng=rng,
        )
        for key, value in local_stats.items():
            stats[key] += int(value)
        for rank, edge in enumerate(edges):
            accepted = False
            reason = ""
            edge_diag: dict[str, Any] = {}
            object_a = str(edge["object_a"])
            object_b = str(edge["object_b"])
            if uf.find(object_a) != uf.find(object_b):
                uf.union(object_a, object_b)
                accepted = True
                stats["accepted_edge_count"] += 1
                a_stats = _object_gt_stats(scene, by_object[object_a], cache)
                b_stats = _object_gt_stats(scene, by_object[object_b], cache)
                u_stats = _union_gt_stats(scene, by_object[object_a], by_object[object_b], cache)
                a_gt = int(a_stats["primary_gt_id"])
                b_gt = int(b_stats["primary_gt_id"])
                same_gt = bool(a_gt > 0 and a_gt == b_gt)
                diff_gt = bool(a_gt > 0 and b_gt > 0 and a_gt != b_gt)
                same_semantic = _semantic_bucket(by_object[object_a]) == _semantic_bucket(by_object[object_b])
                if a_gt > 0 and b_gt > 0:
                    stats["gt_known_connection_count_diagnostic"] += 1
                    stats["same_GT_connection_count_diagnostic"] += int(same_gt)
                    stats["accepted_diff_gt_edge_count_diagnostic"] += int(diff_gt)
                    if same_semantic:
                        stats["same_semantic_known_count_diagnostic"] += 1
                        stats["same_semantic_diff_gt_count_diagnostic"] += int(diff_gt)
                best_iou = max(float(a_stats["primary_gt_iou"]), float(b_stats["primary_gt_iou"]))
                union_iou = float(u_stats["primary_gt_iou"])
                union_minus_best = union_iou - best_iou
                stats["best_pred_IoU_values"].append(best_iou)
                stats["union_pred_IoU_values"].append(union_iou)
                stats["union_minus_best_IoU_values"].append(union_minus_best)
                edge_diag = {
                    "object_a_primary_gt_id_diagnostic": a_gt,
                    "object_b_primary_gt_id_diagnostic": b_gt,
                    "union_primary_gt_id_diagnostic": int(u_stats["primary_gt_id"]),
                    "same_GT_connection_diagnostic": same_gt,
                    "diff_GT_false_connection_diagnostic": diff_gt,
                    "same_semantic_pair_diagnostic": same_semantic,
                    "same_semantic_diff_GT_false_connection_diagnostic": bool(diff_gt and same_semantic),
                    "best_pred_IoU_diagnostic": best_iou,
                    "union_pred_IoU_diagnostic": union_iou,
                    "union_minus_best_IoU_diagnostic": union_minus_best,
                }
            edge_rows.append(
                {
                    "schema_version": "stream4d_v103_r7_phase5_edge_row_v1",
                    "phase_id": PHASE_ID,
                    "variant_id": str(variant["variant_id"]),
                    "control_role": control_role,
                    "scene_id": scene,
                    "edge_rank": int(rank),
                    "object_a": object_a,
                    "object_b": object_b,
                    "anchor_seed_object_a": str(edge["anchor_seed_object_a"]),
                    "anchor_seed_object_c": str(edge["anchor_seed_object_c"]),
                    "support_hub_object": str(edge["support_hub_object"]),
                    "hop_type": str(edge["hop_type"]),
                    "anchor_affinity": float(edge["anchor_affinity"]),
                    "support_affinity": float(edge["support_affinity"]),
                    "support_affinity_second_hop": float(edge["support_affinity_second_hop"]),
                    "support_density_score": float(edge["support_density_score"]),
                    "combined_score": float(edge["combined_score"]),
                    "accepted_union": bool(accepted),
                    "reject_reason": reason,
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
            base_score = float(comp_rows.get("score", pd.Series([0.0])).max())
            cluster_rows.append(
                {
                    "schema_version": "stream4d_v103_r7_phase5_cluster_row_v1",
                    "phase_id": PHASE_ID,
                    "variant_id": str(variant["variant_id"]),
                    "control_role": control_role,
                    "scene_id": scene,
                    "mv_object_id": object_id,
                    "source_object_ids": json.dumps(members, sort_keys=True),
                    "source_object_count": int(len(members)),
                    "frame_count": int(comp_rows["frame_id"].nunique()),
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
                        "schema_version": "stream4d_v103_r7_phase5_frame_mask_row_v1",
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
                        "node_policy": "r7_anchor_seeded_support_propagation",
                        "emit_policy": "component_wta_by_f2_score_support",
                        "readout_mode": "r7_anchor_seeded_support_propagation",
                        "uses_gt_for_prediction": False,
                        "uses_future": False,
                    }
                )
    known = int(stats["gt_known_connection_count_diagnostic"])
    sem_known = int(stats["same_semantic_known_count_diagnostic"])
    stats["same_GT_connection_rate_diagnostic"] = float(stats["same_GT_connection_count_diagnostic"]) / max(known, 1)
    stats["diff_GT_false_connection_rate_diagnostic"] = float(stats["accepted_diff_gt_edge_count_diagnostic"]) / max(known, 1)
    stats["same_semantic_diff_GT_false_connection_rate"] = float(stats["same_semantic_diff_gt_count_diagnostic"]) / max(sem_known, 1)
    for src, dst in [
        ("best_pred_IoU_values", "best_pred_IoU"),
        ("union_pred_IoU_values", "union_pred_IoU"),
        ("union_minus_best_IoU_values", "union_minus_best_IoU"),
    ]:
        vals = [float(v) for v in stats.pop(src)]
        stats[dst] = float(np.mean(vals)) if vals else 0.0
    return scene_rows, edge_rows, cluster_rows, stats


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
    anchor_root = _project(args.anchor_feature_root)
    support_root = _project(args.support_feature_root)
    phase2_summaries = {
        "scene0011_00": _read_json(_project(args.scene0011_phase2_root) / "summary.json"),
        "scene0050_00": _read_json(_project(args.scene0050_phase2_root) / "summary.json"),
    }
    anchor_payloads = {scene: _load_phase5_scene(anchor_root, scene) for scene in phase2_summaries}
    support_payloads = {scene: _load_phase5_scene(support_root, scene) for scene in phase2_summaries}
    base = _adapt_f2_rows(
        f2_root=f2_root,
        phase2_summaries=phase2_summaries,
        phase5_payloads=support_payloads,
        dataset_split=str(args.dataset_split),
        chunk_id=str(args.chunk_id),
    )
    _, anchor_features = _object_tables(base, anchor_payloads)
    _, support_features = _object_tables(base, support_payloads)

    forbidden_tokens = ["da3", "3dgs", "gaussian", "phase9n", "phase9b", "phase9c", "phase9d", "da3_pair"]
    inspected_paths = [phase1_root, f2_root, anchor_root, support_root, _project(args.scene0011_phase2_root), _project(args.scene0050_phase2_root)]
    forbidden_hits = [_rel(path) for path in inspected_paths if any(token in _rel(path).lower() for token in forbidden_tokens)]

    artifact_rows = [
        _artifact_row("r7_phase1_summary", phase1_root / "summary.json"),
        _artifact_row("f2_replay_rows", f2_root / "mv_object_frame_mask_rows.parquet"),
        _artifact_row("anchor_feature_root", anchor_root, note="R6F0 D4RT-only anchor feature root"),
        _artifact_row("support_feature_root", support_root, note="R6F2 D4RT-only support feature root"),
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
        for control_role in ["real", "shuffled_support_control", "density_control"]:
            scene_rows, edge_rows, cluster_rows, stats = _materialize_variant(
                base=base,
                anchor_features=anchor_features,
                support_features=support_features,
                variant=variant,
                chunk_id=str(args.chunk_id),
                control_role=control_role,
                phase2_summaries=phase2_summaries,
            )
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
                "schema_version": "stream4d_v103_r7_phase5_metric_row_v1",
                "phase_id": PHASE_ID,
                "variant_id": eval_variant_id,
                "base_variant_id": str(variant["variant_id"]),
                "variant_family": "R7SP_anchor_seeded_support_propagation",
                "control_role": control_role,
                "hop_mode": str(variant.get("hop_mode", "onehop")),
                "eligible_for_full_dev": bool(variant.get("eligible_for_full_dev", True)) and str(variant.get("hop_mode", "onehop")) == "onehop",
                "anchor_threshold": float(variant["anchor_threshold"]),
                "support_threshold": float(variant["support_threshold"]),
                "anchor_seed_edge_count": int(stats["anchor_seed_edge_count"]),
                "propagated_edge_count": int(stats["propagated_edge_count"]),
                "onehop_edge_count": int(stats["onehop_edge_count"]),
                "twohop_edge_count": int(stats["twohop_edge_count"]),
                "propagated_edge_veto_reject_count": int(stats["propagated_edge_veto_reject_count"]),
                "propagated_edge_semantic_reject_count": int(stats["propagated_edge_semantic_reject_count"]),
                "accepted_S_only_edge_count": int(stats["accepted_S_only_edge_count"]),
                "accepted_edge_count": int(stats["accepted_edge_count"]),
                "accepted_diff_gt_edge_count_diagnostic": int(stats["accepted_diff_gt_edge_count_diagnostic"]),
                "same_GT_connection_rate_diagnostic": float(stats["same_GT_connection_rate_diagnostic"]),
                "diff_GT_false_connection_rate_diagnostic": float(stats["diff_GT_false_connection_rate_diagnostic"]),
                "same_semantic_diff_GT_false_connection_rate": float(stats["same_semantic_diff_GT_false_connection_rate"]),
                "GT_fragment_count_ge2_rate": float(np.mean([float(w.get("gt_fragment_count_ge2_rate", 0.0)) for w in window_rows])) if window_rows else 0.0,
                "best_pred_IoU": float(stats["best_pred_IoU"]),
                "union_pred_IoU": float(stats["union_pred_IoU"]),
                "union_minus_best_IoU": float(stats["union_minus_best_IoU"]),
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
            elif bool(row["eligible_for_full_dev"]) and (best_real is None or float(row["MV_AP_window"]) > float(best_real["MV_AP_window"])):
                best_real = row
            variant_rows.append(
                {
                    "schema_version": "stream4d_v103_r7_phase5_variant_row_v1",
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
    diff_gt_reference_available = "D9_reference_accepted_diff_gt_edge_count_diagnostic" in phase1_summary
    d9_diff_ref = int(_num(phase1_summary.get("D9_reference_accepted_diff_gt_edge_count_diagnostic"), -1))
    same_semantic_ref_available = "D9_reference_same_semantic_diff_GT_false_connection_rate" in phase1_summary
    same_semantic_ref = _num(phase1_summary.get("D9_reference_same_semantic_diff_GT_false_connection_rate"), 0.0)
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
    subset_gate_pass = bool(
        best_real_full
        and not forbidden_hits
        and _num(best_real_full.get("MV_AP_window"), 0.0) >= replay_mv + 0.005
        and _num(best_real_full.get("MV_AP50_window"), 0.0) >= replay_ap50 + 0.010
        and _num(best_real_full.get("real_minus_best_control_MV_AP_window"), 0.0) >= 0.003
        and _num(best_real_full.get("real_minus_best_control_MV_AP50_window"), 0.0) >= 0.006
        and int(best_real_full.get("accepted_S_only_edge_count", 1)) == 0
        and diff_gt_reference_available
        and int(best_real_full.get("accepted_diff_gt_edge_count_diagnostic", 999999)) <= d9_diff_ref
        and same_semantic_ref_available
        and _num(best_real_full.get("same_semantic_diff_GT_false_connection_rate"), 1.0) <= same_semantic_ref
        and int(best_real_full.get("same_frame_collision_count", 1)) == 0
        and int(best_real_full.get("missing_mask_raster_count", 1)) == 0
        and float(best_real_full.get("pixel_collision_rate", 1.0)) == 0.0
    )

    gate_rows = [
        _gate_row("phase1_d4rt_only", not bool(phase1_summary.get("DA3_USED", True)), phase1_summary.get("DA3_USED", ""), False, "Repair R7-1 D4RT-only boundary."),
        _gate_row("no_forbidden_artifact_path_tokens", not forbidden_hits, forbidden_hits, [], "Remove DA3/3DGS/Gaussian/phase9 provider artifacts from R7-5 inputs."),
        _gate_row("accepted_S_only_edge_count_zero", all(int(r.get("accepted_S_only_edge_count", 0)) == 0 for r in metric_rows), 0, 0, "Disallow support-only union."),
        _gate_row("best_MV_AP_window_ge_replay_plus_0p005", _num(best_real_full.get("MV_AP_window"), 0.0) >= replay_mv + 0.005, best_real_full.get("MV_AP_window", ""), replay_mv + 0.005, "If false, inspect anchor seed sparsity and veto threshold."),
        _gate_row("best_MV_AP50_window_ge_replay_plus_0p010", _num(best_real_full.get("MV_AP50_window"), 0.0) >= replay_ap50 + 0.010, best_real_full.get("MV_AP50_window", ""), replay_ap50 + 0.010, "If false, do not promote."),
        _gate_row("real_minus_best_control_MV_AP_window_ge_0p003", _num(best_real_full.get("real_minus_best_control_MV_AP_window"), 0.0) >= 0.003, best_real_full.get("real_minus_best_control_MV_AP_window", ""), 0.003, "Control explains gain if false."),
        _gate_row("real_minus_best_control_MV_AP50_window_ge_0p006", _num(best_real_full.get("real_minus_best_control_MV_AP50_window"), 0.0) >= 0.006, best_real_full.get("real_minus_best_control_MV_AP50_window", ""), 0.006, "Control explains AP50 gain if false."),
        _gate_row("accepted_diff_gt_reference_available", diff_gt_reference_available, sorted(phase1_summary.keys()), "D9_reference_accepted_diff_gt_edge_count_diagnostic", "Add exact edge diagnostic reference before formal promotion."),
        _gate_row("same_semantic_diff_gt_reference_available", same_semantic_ref_available, sorted(phase1_summary.keys()), "D9_reference_same_semantic_diff_GT_false_connection_rate", "Add exact edge diagnostic reference before formal promotion."),
        _gate_row("collision_missing_safe", int(best_real_full.get("same_frame_collision_count", 1)) == 0 and int(best_real_full.get("missing_mask_raster_count", 1)) == 0 and float(best_real_full.get("pixel_collision_rate", 1.0)) == 0.0, best_real_full, "collision=0 missing=0 pixel_collision_rate=0", "Repair materialization if false."),
    ]
    failure_rows: list[dict[str, Any]] = []
    if forbidden_hits:
        failure_rows.append(_failure_row("R7_DA3_ARTIFACT_LEAKAGE", {"forbidden_hits": forbidden_hits}, "Terminate contaminated variant and rerun with D4RT-only inputs."))
    if not subset_gate_pass:
        failure_rows.append(
            _failure_row(
                "NO_GO_R7_5_ANCHOR_SEEDED_SUPPORT_PROPAGATION_SUBSET_GATE",
                {"best_real": best_real_full, "replay_MV_AP_window": replay_mv, "replay_MV_AP50_window": replay_ap50},
                "Proceed to R7-7 local evaluation/inconsistency or design a stricter object-specific seed filter; do not promote R7SP.",
            )
        )
    else:
        failure_rows.append(_failure_row("NONE", "R7-5 subset gate passed.", "Freeze config before full-dev.", severity="info"))

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
        "schema_version": "stream4d_v103_r7_phase5_summary_v1",
        "phase": "R7-5",
        "phase_id": PHASE_ID,
        "phase_pass": subset_gate_pass,
        "decision": "PASS_R7_5_SUBSET_CANDIDATE" if subset_gate_pass else "NO_GO_R7_5_ANCHOR_SEEDED_SUPPORT_PROPAGATION",
        "runtime_sec": time.time() - t0,
        "variant_count": len(VARIANTS),
        "metric_row_count": len(metric_rows),
        "edge_row_count": len(all_edge_rows),
        "best_real_variant_id": best_real_full.get("variant_id", ""),
        "best_real_MV_AP_window": best_real_full.get("MV_AP_window", ""),
        "best_real_MV_AP50_window": best_real_full.get("MV_AP50_window", ""),
        "best_real_minus_replay_MV_AP_window": _num(best_real_full.get("MV_AP_window"), 0.0) - replay_mv if best_real_full else "",
        "best_real_minus_replay_MV_AP50_window": _num(best_real_full.get("MV_AP50_window"), 0.0) - replay_ap50 if best_real_full else "",
        "best_real_minus_best_control_MV_AP_window": best_real_full.get("real_minus_best_control_MV_AP_window", ""),
        "best_real_minus_best_control_MV_AP50_window": best_real_full.get("real_minus_best_control_MV_AP50_window", ""),
        "anchor_seed_edge_count": best_real_full.get("anchor_seed_edge_count", ""),
        "propagated_edge_count": best_real_full.get("propagated_edge_count", ""),
        "onehop_edge_count": best_real_full.get("onehop_edge_count", ""),
        "twohop_edge_count": best_real_full.get("twohop_edge_count", ""),
        "accepted_S_only_edge_count": best_real_full.get("accepted_S_only_edge_count", ""),
        "accepted_diff_gt_edge_count_diagnostic": best_real_full.get("accepted_diff_gt_edge_count_diagnostic", ""),
        "same_GT_connection_rate_diagnostic": best_real_full.get("same_GT_connection_rate_diagnostic", ""),
        "same_semantic_diff_GT_false_connection_rate": best_real_full.get("same_semantic_diff_GT_false_connection_rate", ""),
        "GT_fragment_count_ge2_rate": best_real_full.get("GT_fragment_count_ge2_rate", ""),
        "best_pred_IoU": best_real_full.get("best_pred_IoU", ""),
        "union_pred_IoU": best_real_full.get("union_pred_IoU", ""),
        "union_minus_best_IoU": best_real_full.get("union_minus_best_IoU", ""),
        "same_frame_collision_count": best_real_full.get("same_frame_collision_count", ""),
        "pixel_collision_rate": best_real_full.get("pixel_collision_rate", ""),
        "missing_mask_raster_count": best_real_full.get("missing_mask_raster_count", ""),
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
        "truthfulness_note": "R7-5 starts from A_anchor seed edges and allows D4RT-only S_support one-hop propagation; support-only full-graph broadcast is never allowed. GT is used only for evaluator/diagnostics.",
    }
    _write_json(out / "summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build v103 R7-5 D4RT-only anchor-seeded support propagation variants.")
    parser.add_argument("--output-root", default=str(DEFAULT_OUT))
    parser.add_argument("--r7-phase1-root", default=str(DEFAULT_R7_PHASE1_ROOT))
    parser.add_argument("--f2-root", default=str(DEFAULT_F2_ROOT))
    parser.add_argument("--anchor-feature-root", default=str(DEFAULT_ANCHOR_FEATURE))
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
