#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


STREAM3D_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = STREAM3D_ROOT.parent
AUDIT_ROOT = STREAM3D_ROOT / "outputs" / "audit"
TOOLS_DIR = Path(__file__).resolve().parent
if str(STREAM3D_ROOT) not in sys.path:
    sys.path.insert(0, str(STREAM3D_ROOT))
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from build_v103_phase7_causal_history_token_readiness import (  # noqa: E402
    _load_mask,
)


PHASE_ID = "v103_object_specific_carrier_support_diagnostic"
SCHEMA_PREFIX = "stream4d_v103_object_specific_support"
DEFAULT_OUT = AUDIT_ROOT / "v103_object_specific_carrier_support_purityfirst_r1"
DEFAULT_S1_ROOT = AUDIT_ROOT / "v103_supp_phaseS1_multirole_carriers_purityfirst_r1"
DEFAULT_S3_ROOT = AUDIT_ROOT / "v103_supp_phaseS3_scaffolded_mask_graph_purityfirst_directpair_c0001_r2"
DEFAULT_S4_ROOT = AUDIT_ROOT / "v103_supp_phaseS4_post_birth_history_inheritance_purityfirst_s3v1_r1"
DEFAULT_SCENE0011_PHASE2 = (
    AUDIT_ROOT / "v103_phase2_stratified_q5c_objlike16384_scene0011_c0001_cap24576_qchunk16384_d4rt48mix_maskbalanced8"
)
DEFAULT_SCENE0050_PHASE2 = (
    AUDIT_ROOT / "v103_phase2_stratified_q5c_objlike16384_scene0050_c0001_cap24576_qchunk16384_d4rt48mix_maskbalanced8"
)
MMAP_BATCH_KEYS = ("carrier_id", "uv_pred", "valid", "visibility_prob", "confidence_prob", "xyz_ref")


def _project(path: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    if p.parts and p.parts[0] == "Stream3D":
        return REPO_ROOT / p
    return STREAM3D_ROOT / p


def _rel(path: str | Path) -> str:
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


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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


def _num(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def _truth(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _load_phase2_scene_fast(root: Path) -> tuple[dict[str, Any], np.ndarray, dict[str, Any], str]:
    summary = _read_json(root / "summary.json")
    frame_ids = [int(v) for v in summary["frame_ids"]]
    mask_root = _project(summary["mask_root"])
    masks = np.stack([_load_mask(mask_root / f"{frame_id}.png") for frame_id in frame_ids], axis=0)
    cache_dir = root / "carrier_batch_mmap_cache"
    if cache_dir.exists() and all((cache_dir / f"{key}.npy").exists() for key in MMAP_BATCH_KEYS):
        batch = {key: np.load(cache_dir / f"{key}.npy", mmap_mode="r") for key in MMAP_BATCH_KEYS}
        return summary, masks, batch, "phase2_mmap_cache"
    batch_npz = np.load(root / "carrier_batch.npz", allow_pickle=False)
    batch = {key: batch_npz[key] for key in MMAP_BATCH_KEYS}
    return summary, masks, batch, "compressed_npz_fallback"


def _project_labels_for_indices_fast(
    *,
    batch: dict[str, Any],
    masks: np.ndarray,
    carrier_indices: np.ndarray,
    batch_backend: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str, float]:
    t0 = time.time()
    frame_count = int(batch["valid"].shape[0])
    height, width = masks.shape[1:]
    n = int(carrier_indices.shape[0])
    labels = np.full((frame_count, n), -1, dtype=np.int32)
    ok = np.zeros((frame_count, n), dtype=bool)
    weights = np.zeros((frame_count, n), dtype=np.float32)
    for fi in range(frame_count):
        uv = np.asarray(batch["uv_pred"][fi, carrier_indices], dtype=np.float32)
        valid = np.asarray(batch["valid"][fi, carrier_indices], dtype=bool)
        visibility = np.asarray(batch["visibility_prob"][fi, carrier_indices], dtype=np.float32)
        confidence = np.asarray(batch["confidence_prob"][fi, carrier_indices], dtype=np.float32)
        xyz = np.asarray(batch["xyz_ref"][fi, carrier_indices], dtype=np.float32)
        finite = np.isfinite(uv).all(axis=1) & np.isfinite(xyz).all(axis=1)
        in_img = valid & finite & (uv[:, 0] >= 0.0) & (uv[:, 0] <= 1.0) & (uv[:, 1] >= 0.0) & (uv[:, 1] <= 1.0)
        xs = np.rint(np.clip(uv[:, 0], 0.0, 1.0) * float(max(width - 1, 1))).astype(np.int32)
        ys = np.rint(np.clip(uv[:, 1], 0.0, 1.0) * float(max(height - 1, 1))).astype(np.int32)
        labels[fi, in_img] = masks[fi, ys[in_img], xs[in_img]]
        ok[fi] = in_img
        weights[fi] = np.where(in_img, visibility * confidence, 0.0).astype(np.float32, copy=False)
    return labels, ok, weights, f"numpy_{batch_backend}_framewise_projection", time.time() - t0


def _scene_phase2_roots(args: argparse.Namespace) -> dict[str, Path]:
    return {
        "scene0011_00": _project(args.scene0011_phase2_root),
        "scene0050_00": _project(args.scene0050_phase2_root),
    }


def _carrier_indices_for_role(scene_roles: pd.DataFrame, batch_carrier_ids: np.ndarray, role_name: str) -> tuple[np.ndarray, np.ndarray, bool]:
    role_mask = scene_roles["role_name"].astype(str).to_numpy() == str(role_name)
    role_carrier_ids = scene_roles.loc[role_mask, "carrier_id"].to_numpy(dtype=np.int64, copy=False)
    all_role_ids = scene_roles["carrier_id"].to_numpy(dtype=np.int64, copy=False)
    if all_role_ids.shape[0] == batch_carrier_ids.shape[0] and np.array_equal(all_role_ids, batch_carrier_ids):
        return np.flatnonzero(role_mask).astype(np.int64, copy=False), role_carrier_ids, True
    lookup = pd.Series(np.arange(batch_carrier_ids.shape[0], dtype=np.int64), index=batch_carrier_ids)
    mapped = lookup.reindex(role_carrier_ids)
    if mapped.isna().any():
        missing = int(mapped.isna().sum())
        raise RuntimeError(f"{missing} {role_name} carrier ids are missing from Phase2 batch carrier_id")
    return mapped.to_numpy(dtype=np.int64, copy=False), role_carrier_ids, False


def _current_object_rows(phaseS3_root: Path, variant_id: str, chunk_id: str) -> pd.DataFrame:
    rows = pd.read_csv(phaseS3_root / "object_frame_mask_rows.csv")
    uses_gt = rows["uses_gt_for_prediction"].map(_truth)
    uses_future = rows["uses_future"].map(_truth)
    rows = rows[
        (rows["variant_id"].astype(str) == str(variant_id))
        & (rows["chunk_id"].astype(str) == str(chunk_id))
        & (~uses_gt)
        & (~uses_future)
    ].copy()
    if rows.empty:
        raise RuntimeError(f"no current S3 object rows for variant_id={variant_id} chunk_id={chunk_id}")
    return rows


def _object_row_base(scene_rows: pd.DataFrame) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for oid, group in scene_rows.groupby("mv_object_id", sort=True):
        frame_ids = group["frame_id"].astype(int).to_numpy()
        out[str(oid)] = {
            "source_local_object_id": str(oid),
            "row_count": int(len(group)),
            "frame_count": int(len(np.unique(frame_ids))),
            "frame_id_min": int(np.min(frame_ids)) if frame_ids.size else "",
            "frame_id_max": int(np.max(frame_ids)) if frame_ids.size else "",
            "object_score": float(group["object_score"].astype(float).max()) if "object_score" in group else 0.0,
            "selected_mask_area_mean": float(group.get("selected_mask_area", pd.Series(dtype=float)).astype(float).mean())
            if "selected_mask_area" in group
            else 0.0,
        }
    return out


def _compute_role_hits(
    *,
    scene: str,
    role_name: str,
    labels: np.ndarray,
    ok: np.ndarray,
    weights: np.ndarray,
    role_carrier_ids: np.ndarray,
    scene_rows: pd.DataFrame,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    object_ids = sorted(scene_rows["mv_object_id"].astype(str).unique().tolist())
    obj_index = {oid: i for i, oid in enumerate(object_ids)}
    owner_by_frame: dict[int, dict[int, list[int]]] = defaultdict(lambda: defaultdict(list))
    row_keys_by_obj: dict[int, set[tuple[int, int]]] = defaultdict(set)
    for row in scene_rows.to_dict("records"):
        oid = str(row["mv_object_id"])
        oi = obj_index[oid]
        key = (int(row["frame_local_index"]), int(row["selected_mask_id"]))
        owner_by_frame[key[0]][key[1]].append(oi)
        row_keys_by_obj[oi].add(key)

    hit_count = np.zeros((len(object_ids),), dtype=np.int64)
    weight_sum = np.zeros((len(object_ids),), dtype=np.float64)
    hit_row_keys: list[set[tuple[int, int]]] = [set() for _ in object_ids]
    carrier_sets: list[set[int]] = [set() for _ in object_ids]
    carrier_hit_any = np.zeros((role_carrier_ids.shape[0],), dtype=bool)

    for fi in range(labels.shape[0]):
        frame_owner = owner_by_frame.get(int(fi), {})
        if not frame_owner:
            continue
        lab = labels[fi]
        good = ok[fi] & (lab > 0) & np.isfinite(weights[fi]) & (weights[fi] > 0)
        if not np.any(good):
            continue
        good_idx = np.flatnonzero(good).astype(np.int64, copy=False)
        good_lab = lab[good_idx]
        for mask_id, targets in frame_owner.items():
            if not targets:
                continue
            local = good_lab == int(mask_id)
            if not np.any(local):
                continue
            idxs = good_idx[local]
            w = np.asarray(weights[fi, idxs], dtype=np.float32)
            ids = role_carrier_ids[idxs]
            for oi in targets:
                hit_count[oi] += int(idxs.shape[0])
                weight_sum[oi] += float(np.sum(w))
                hit_row_keys[oi].add((int(fi), int(mask_id)))
                carrier_sets[oi].update(int(v) for v in ids.tolist())
            carrier_hit_any[idxs] = True

    object_rows: dict[str, dict[str, Any]] = {}
    for oid, oi in obj_index.items():
        row_count = max(len(row_keys_by_obj[oi]), 1)
        object_rows[oid] = {
            f"{role_name}_hit_count": int(hit_count[oi]),
            f"{role_name}_unique_carrier_count": int(len(carrier_sets[oi])),
            f"{role_name}_support_weight_sum": float(weight_sum[oi]),
            f"{role_name}_hit_row_count": int(len(hit_row_keys[oi])),
            f"{role_name}_hit_row_rate": float(len(hit_row_keys[oi]) / row_count),
        }
    meta = {
        "scene_id": scene,
        "role_name": role_name,
        "role_carrier_count": int(role_carrier_ids.shape[0]),
        "role_carrier_current_object_hit_rate": float(np.mean(carrier_hit_any)) if carrier_hit_any.size else 0.0,
        "object_with_role_hit_rate": float(np.mean([object_rows[oid][f"{role_name}_unique_carrier_count"] > 0 for oid in object_ids]))
        if object_ids
        else 0.0,
        "object_role_unique_carrier_count_p10": float(
            np.percentile([object_rows[oid][f"{role_name}_unique_carrier_count"] for oid in object_ids], 10)
        )
        if object_ids
        else 0.0,
        "object_role_unique_carrier_count_median": float(
            np.percentile([object_rows[oid][f"{role_name}_unique_carrier_count"] for oid in object_ids], 50)
        )
        if object_ids
        else 0.0,
    }
    return object_rows, meta


def _load_edge_stats(phaseS3_root: Path, variant_id: str) -> dict[str, dict[str, int]]:
    cluster_path = phaseS3_root / "cluster_rows.parquet"
    edge_path = phaseS3_root / "edge_intervention_rows.parquet"
    if not cluster_path.exists() or not edge_path.exists():
        return {}
    clusters = pd.read_parquet(cluster_path)
    clusters = clusters[clusters["variant_id"].astype(str) == str(variant_id)].copy()
    source_to_cluster: dict[str, str] = {}
    for row in clusters.to_dict("records"):
        try:
            sources = json.loads(str(row["source_object_ids"]))
        except Exception:
            sources = []
        for src in sources:
            source_to_cluster[str(src)] = str(row["mv_object_id"])

    stats: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    edges = pd.read_parquet(edge_path)
    edges = edges[edges["variant_id"].astype(str) == str(variant_id)].copy()
    for row in edges.to_dict("records"):
        for key in ["object_a", "object_b"]:
            cluster_id = source_to_cluster.get(str(row.get(key, "")))
            if not cluster_id:
                continue
            stats[cluster_id]["edge_endpoint_count"] += 1
            if _truth(row.get("positive_candidate", False)):
                stats[cluster_id]["positive_candidate_edge_endpoint_count"] += 1
            if _truth(row.get("accepted_union", False)):
                stats[cluster_id]["accepted_union_edge_endpoint_count"] += 1
            if _truth(row.get("veto_hard_cannot_link", False)):
                stats[cluster_id]["veto_hard_cannot_link_endpoint_count"] += 1
            if str(row.get("reject_reason", "")):
                stats[cluster_id]["rejected_edge_endpoint_count"] += 1
            if _truth(row.get("direct_pair_positive", False)):
                stats[cluster_id]["direct_pair_positive_endpoint_count"] += 1
    return {key: dict(value) for key, value in stats.items()}


def _load_history_assignment(phaseS4_root: Path, variant_id: str) -> dict[str, dict[str, Any]]:
    path = phaseS4_root / "history_assignment_rows.parquet"
    if not path.exists():
        return {}
    rows = pd.read_parquet(path)
    rows = rows[rows["variant_id"].astype(str) == str(variant_id)].copy()
    out: dict[str, dict[str, Any]] = {}
    for row in rows.to_dict("records"):
        oid = str(row["source_local_object_id"])
        out[oid] = {
            "history_variant_id": str(variant_id),
            "assigned_history_id": str(row.get("assigned_history_id", "")),
            "assigned_before_one_to_one": _truth(row.get("assigned_before_one_to_one", False)),
            "assigned_after_one_to_one": _truth(row.get("assigned_after_one_to_one", False)),
            "top1_history_id": str(row.get("top1_history_id", "")),
            "object_history_top1_score": float(row.get("object_history_top1_score", 0.0)),
            "object_history_top1_margin": float(row.get("object_history_top1_margin", 0.0)),
            "object_history_entropy": float(row.get("object_history_entropy", 1.0)),
            "support_weight_sum": float(row.get("support_weight_sum", 0.0)),
            "carrier_hit_count": int(row.get("carrier_hit_count", 0)),
            "assigned_carrier_hit_count": int(row.get("assigned_carrier_hit_count", 0)),
            "unique_carrier_count": int(row.get("unique_carrier_count", 0)),
            "reject_reason": str(row.get("reject_reason", "")),
        }
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Diagnose object-specific carrier support for v103 S1/S3/S4.")
    parser.add_argument("--output-root", default=str(DEFAULT_OUT))
    parser.add_argument("--phaseS1-root", default=str(DEFAULT_S1_ROOT))
    parser.add_argument("--phaseS3-root", default=str(DEFAULT_S3_ROOT))
    parser.add_argument("--phaseS4-root", default=str(DEFAULT_S4_ROOT))
    parser.add_argument("--scene0011-phase2-root", default=str(DEFAULT_SCENE0011_PHASE2))
    parser.add_argument("--scene0050-phase2-root", default=str(DEFAULT_SCENE0050_PHASE2))
    parser.add_argument("--phaseS3-variant-id", default="S3_V1_anchor_positive_only")
    parser.add_argument("--phaseS4-variant-id", default="S4_H0_real_strict_post_birth_inheritance")
    parser.add_argument("--chunk-id", default="c0001")
    parser.add_argument("--cupy-device-id", type=int, default=0)
    parser.add_argument("--anchor-object-hit-rate-floor", type=float, default=0.50)
    parser.add_argument("--edge-object-hit-rate-floor", type=float, default=0.25)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    started = time.time()
    out = _project(args.output_root)
    out.mkdir(parents=True, exist_ok=True)
    (out / "last_command.txt").write_text(" ".join([sys.executable, *sys.argv]) + "\n", encoding="utf-8")

    phaseS1_root = _project(args.phaseS1_root)
    phaseS3_root = _project(args.phaseS3_root)
    phaseS4_root = _project(args.phaseS4_root)
    s3_summary = _read_json(phaseS3_root / "summary.json")
    s4_summary = _read_json(phaseS4_root / "summary.json")
    role_rows = pd.read_parquet(phaseS1_root / "carrier_role_rows.parquet")
    current_rows = _current_object_rows(phaseS3_root, str(args.phaseS3_variant_id), str(args.chunk_id))
    edge_stats = _load_edge_stats(phaseS3_root, str(args.phaseS3_variant_id))
    history_assignment = _load_history_assignment(phaseS4_root, str(args.phaseS4_variant_id))

    object_rows: list[dict[str, Any]] = []
    scene_rows: list[dict[str, Any]] = []
    projection_rows: list[dict[str, Any]] = []
    scene_phase2_roots = _scene_phase2_roots(args)

    for scene, phase2_root in scene_phase2_roots.items():
        print(f"[{PHASE_ID}] scene={scene} load inputs", file=sys.stderr, flush=True)
        scene_current = current_rows[current_rows["scene_id"].astype(str) == scene].copy()
        if scene_current.empty:
            continue
        scene_roles = role_rows[role_rows["scene_id"].astype(str) == scene].reset_index(drop=True)
        summary, masks, batch, batch_backend = _load_phase2_scene_fast(phase2_root)
        batch_carrier_ids = np.asarray(batch["carrier_id"], dtype=np.int64)
        base = _object_row_base(scene_current)
        role_metas: list[dict[str, Any]] = []

        for role_name in ["A_anchor", "S_support"]:
            role_indices, role_carrier_ids, positional_match = _carrier_indices_for_role(scene_roles, batch_carrier_ids, role_name)
            print(
                f"[{PHASE_ID}] scene={scene} role={role_name} project carriers={role_carrier_ids.shape[0]}",
                file=sys.stderr,
                flush=True,
            )
            labels, ok, weights, backend, runtime = _project_labels_for_indices_fast(
                batch=batch,
                masks=masks,
                carrier_indices=role_indices,
                batch_backend=batch_backend,
            )
            role_object_rows, role_meta = _compute_role_hits(
                scene=scene,
                role_name=role_name,
                labels=labels,
                ok=ok,
                weights=weights,
                role_carrier_ids=role_carrier_ids,
                scene_rows=scene_current,
            )
            print(
                f"[{PHASE_ID}] scene={scene} role={role_name} projected backend={backend} runtime_sec={runtime:.3f}",
                file=sys.stderr,
                flush=True,
            )
            role_meta.update(
                {
                    "schema_version": f"{SCHEMA_PREFIX}_projection_row_v1",
                    "phase_id": PHASE_ID,
                    "phase2_root": _rel(phase2_root),
                    "projection_backend": backend,
                    "projection_runtime_sec": float(runtime),
                    "carrier_id_positional_match": bool(positional_match),
                    "uses_gt_for_prediction": False,
                }
            )
            role_metas.append(role_meta)
            projection_rows.append(role_meta)
            for oid, metrics in role_object_rows.items():
                base.setdefault(oid, {"source_local_object_id": oid}).update(metrics)

        for oid, row in base.items():
            hist = history_assignment.get(oid, {})
            edges = edge_stats.get(oid, {})
            merged = {
                "schema_version": f"{SCHEMA_PREFIX}_object_row_v1",
                "phase_id": PHASE_ID,
                "scene_id": scene,
                "phaseS3_variant_id": str(args.phaseS3_variant_id),
                "phaseS4_variant_id": str(args.phaseS4_variant_id),
                **row,
                "A_anchor_unique_carrier_count": int(row.get("A_anchor_unique_carrier_count", 0)),
                "S_support_unique_carrier_count": int(row.get("S_support_unique_carrier_count", 0)),
                "support_only_no_anchor": int(row.get("A_anchor_unique_carrier_count", 0)) == 0
                and int(row.get("S_support_unique_carrier_count", 0)) > 0,
                "no_anchor_no_support": int(row.get("A_anchor_unique_carrier_count", 0)) == 0
                and int(row.get("S_support_unique_carrier_count", 0)) == 0,
                **{
                    "edge_endpoint_count": int(edges.get("edge_endpoint_count", 0)),
                    "positive_candidate_edge_endpoint_count": int(edges.get("positive_candidate_edge_endpoint_count", 0)),
                    "accepted_union_edge_endpoint_count": int(edges.get("accepted_union_edge_endpoint_count", 0)),
                    "veto_hard_cannot_link_endpoint_count": int(edges.get("veto_hard_cannot_link_endpoint_count", 0)),
                    "rejected_edge_endpoint_count": int(edges.get("rejected_edge_endpoint_count", 0)),
                    "direct_pair_positive_endpoint_count": int(edges.get("direct_pair_positive_endpoint_count", 0)),
                },
                **hist,
                "uses_gt_for_prediction": False,
                "uses_future": False,
            }
            object_rows.append(merged)

        scene_obj = [r for r in object_rows if r["scene_id"] == scene]
        object_count = len(scene_obj)
        anchor_hit = np.asarray([int(r.get("A_anchor_unique_carrier_count", 0)) > 0 for r in scene_obj], dtype=bool)
        support_hit = np.asarray([int(r.get("S_support_unique_carrier_count", 0)) > 0 for r in scene_obj], dtype=bool)
        accepted_edge = np.asarray([int(r.get("accepted_union_edge_endpoint_count", 0)) > 0 for r in scene_obj], dtype=bool)
        assigned_after = np.asarray([bool(r.get("assigned_after_one_to_one", False)) for r in scene_obj], dtype=bool)
        assigned_before = np.asarray([bool(r.get("assigned_before_one_to_one", False)) for r in scene_obj], dtype=bool)
        one_to_one_lost = [r for r in scene_obj if str(r.get("reject_reason", "")) == "one_to_one_history_competition_lost"]

        anchor_object_hit_rate = float(np.mean(anchor_hit)) if object_count else 0.0
        support_object_hit_rate = float(np.mean(support_hit)) if object_count else 0.0
        accepted_edge_object_rate = float(np.mean(accepted_edge)) if object_count else 0.0
        history_assigned_after_rate = float(np.mean(assigned_after)) if object_count else 0.0
        blockers: list[str] = []
        if anchor_object_hit_rate < float(args.anchor_object_hit_rate_floor):
            blockers.append("anchor_object_coverage_sparse")
        if support_object_hit_rate - anchor_object_hit_rate >= 0.25:
            blockers.append("support_coverage_not_promotable_without_object_specific_anchor")
        if accepted_edge_object_rate < float(args.edge_object_hit_rate_floor):
            blockers.append("accepted_edge_intervention_sparse")
        if len(one_to_one_lost) > max(1, int(np.count_nonzero(assigned_after))):
            blockers.append("history_one_to_one_competition")
        if (
            _num(s4_summary.get("real_minus_shuffled_MV_AP_scene")) < 0.006
            or _num(s4_summary.get("real_minus_stale_MV_AP_scene")) < 0.006
            or _num(s4_summary.get("real_minus_semantic_MV_AP_scene")) < 0.003
        ):
            blockers.append("history_control_bias")

        scene_rows.append(
            {
                "schema_version": f"{SCHEMA_PREFIX}_scene_row_v1",
                "phase_id": PHASE_ID,
                "scene_id": scene,
                "object_count": object_count,
                "A_anchor_object_hit_rate": anchor_object_hit_rate,
                "S_support_object_hit_rate": support_object_hit_rate,
                "support_only_no_anchor_object_count": int(
                    sum(bool(r.get("support_only_no_anchor", False)) for r in scene_obj)
                ),
                "no_anchor_no_support_object_count": int(sum(bool(r.get("no_anchor_no_support", False)) for r in scene_obj)),
                "accepted_edge_object_rate": accepted_edge_object_rate,
                "history_assigned_before_one_to_one_rate": float(np.mean(assigned_before)) if object_count else 0.0,
                "history_assigned_after_one_to_one_rate": history_assigned_after_rate,
                "one_to_one_history_competition_lost_count": int(len(one_to_one_lost)),
                "s3_best_delta_MV_AP_window": s3_summary.get("best_minus_baseline_MV_AP_window", ""),
                "s4_real_minus_shuffled_MV_AP_scene": s4_summary.get("real_minus_shuffled_MV_AP_scene", ""),
                "s4_real_minus_stale_MV_AP_scene": s4_summary.get("real_minus_stale_MV_AP_scene", ""),
                "s4_real_minus_semantic_MV_AP_scene": s4_summary.get("real_minus_semantic_MV_AP_scene", ""),
                "blocker_tags": ";".join(blockers),
                "uses_gt_for_prediction": False,
                "uses_gt_for_eval": True,
            }
        )

    anchor_sparse = any(
        _num(row.get("A_anchor_object_hit_rate")) < float(args.anchor_object_hit_rate_floor) for row in scene_rows
    )
    edge_sparse = any(
        _num(row.get("accepted_edge_object_rate")) < float(args.edge_object_hit_rate_floor) for row in scene_rows
    )
    history_control_bias = (
        _num(s4_summary.get("real_minus_shuffled_MV_AP_scene")) < 0.006
        or _num(s4_summary.get("real_minus_stale_MV_AP_scene")) < 0.006
        or _num(s4_summary.get("real_minus_semantic_MV_AP_scene")) < 0.003
    )
    gate_rows = [
        {
            "schema_version": f"{SCHEMA_PREFIX}_gate_row_v1",
            "phase_id": PHASE_ID,
            "gate_id": "all_scenes_anchor_object_hit_rate_ge_floor",
            "pass": not anchor_sparse,
            "observed": {row["scene_id"]: row["A_anchor_object_hit_rate"] for row in scene_rows},
            "required": f">= {args.anchor_object_hit_rate_floor}",
            "uses_gt_for_prediction": False,
            "uses_gt_for_eval": False,
        },
        {
            "schema_version": f"{SCHEMA_PREFIX}_gate_row_v1",
            "phase_id": PHASE_ID,
            "gate_id": "all_scenes_accepted_edge_object_rate_ge_floor",
            "pass": not edge_sparse,
            "observed": {row["scene_id"]: row["accepted_edge_object_rate"] for row in scene_rows},
            "required": f">= {args.edge_object_hit_rate_floor}",
            "uses_gt_for_prediction": False,
            "uses_gt_for_eval": False,
        },
        {
            "schema_version": f"{SCHEMA_PREFIX}_gate_row_v1",
            "phase_id": PHASE_ID,
            "gate_id": "s4_history_controls_pass",
            "pass": not history_control_bias,
            "observed": (
                f"real_minus_shuffled={s4_summary.get('real_minus_shuffled_MV_AP_scene', '')}; "
                f"real_minus_stale={s4_summary.get('real_minus_stale_MV_AP_scene', '')}; "
                f"real_minus_semantic={s4_summary.get('real_minus_semantic_MV_AP_scene', '')}"
            ),
            "required": "shuffled/stale >= 0.006; semantic >= 0.003",
            "uses_gt_for_prediction": False,
            "uses_gt_for_eval": True,
        },
    ]

    decision = "OBJECT_SPECIFIC_CARRIER_SUPPORT_BLOCKER_CONFIRMED"
    if not anchor_sparse and not edge_sparse and not history_control_bias:
        decision = "OBJECT_SPECIFIC_SUPPORT_DIAGNOSTIC_PASS_REVIEW_REQUIRED"

    summary = {
        "schema_version": f"{SCHEMA_PREFIX}_summary_v1",
        "phase_id": PHASE_ID,
        "created_unix_time": time.time(),
        "runtime_sec": time.time() - started,
        "decision": decision,
        "failure_count": 0,
        "inputs": {
            "phaseS1_root": phaseS1_root,
            "phaseS3_root": phaseS3_root,
            "phaseS4_root": phaseS4_root,
            "phase2_roots": scene_phase2_roots,
            "phaseS3_variant_id": str(args.phaseS3_variant_id),
            "phaseS4_variant_id": str(args.phaseS4_variant_id),
        },
        "scene_count": len(scene_rows),
        "anchor_sparse": anchor_sparse,
        "edge_sparse": edge_sparse,
        "history_control_bias": history_control_bias,
        "scene_blocker_tags": {row["scene_id"]: row["blocker_tags"] for row in scene_rows},
        "outputs": {
            "object_support_rows": out / "object_support_rows.csv",
            "scene_attribution_rows": out / "scene_attribution_rows.csv",
            "projection_rows": out / "projection_rows.csv",
            "gate_rows": out / "gate_rows.csv",
            "last_command": out / "last_command.txt",
            "summary": out / "summary.json",
        },
        "truthfulness_note": (
            "This diagnostic projects A_anchor and S_support carriers onto current S3 object masks. "
            "It does not recompute AP and does not use GT for prediction; S4 control metrics are read as evaluation evidence only."
        ),
        "uses_gt_for_prediction": False,
        "uses_gt_for_eval": True,
    }

    _write_csv(out / "object_support_rows.csv", object_rows)
    _write_csv(out / "scene_attribution_rows.csv", scene_rows)
    _write_csv(out / "projection_rows.csv", projection_rows)
    _write_csv(out / "gate_rows.csv", gate_rows)
    _write_json(out / "summary.json", summary)
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
