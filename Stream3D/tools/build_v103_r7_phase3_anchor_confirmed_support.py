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
    DEFAULT_PHASE2_SCENE0011,
    DEFAULT_PHASE2_SCENE0050,
    UnionFind,
    _adapt_f2_rows,
    _broad_support_risk,
    _load_phase5_scene,
    _object_tables,
    _specific_conflict,
)


AUDIT_ROOT = STREAM3D_ROOT / "outputs" / "audit"
PHASE_ID = "v103_r7_phase3_anchor_confirmed_support"
DEFAULT_OUT = AUDIT_ROOT / "v103_r7_phase3_anchor_confirmed_support"
DEFAULT_R7_PHASE1_ROOT = AUDIT_ROOT / "v103_r7_phase1_edge_attribution_exact"
DEFAULT_F2_ROOT = AUDIT_ROOT / "v100_phase2c_overlap3_local_repair"
DEFAULT_PHASE5_ROOT = AUDIT_ROOT / "v103_supp_r6_phase2_support_conditioned_local_ap/phase5_like_features"
DEFAULT_ANCHOR_FEATURE = DEFAULT_PHASE5_ROOT / "R6F0_anchor_only_replay"
DEFAULT_SUPPORT_FEATURE = DEFAULT_PHASE5_ROOT / "R6F2_support010_specificity_semantic"
DEFAULT_SCENE0011_PHASE2 = AUDIT_ROOT / "v103_phase2_stratified_q5c_objlike16384_scene0011_c0001_cap24576_qchunk16384_d4rt48mix_maskbalanced8"
DEFAULT_SCENE0050_PHASE2 = AUDIT_ROOT / "v103_phase2_stratified_q5c_objlike16384_scene0050_c0001_cap24576_qchunk16384_d4rt48mix_maskbalanced8"


VARIANTS = [
    {
        "variant_id": "R7AS1_strict_anchor_support_confirm_tauS070",
        "anchor_threshold": 0.90,
        "support_threshold": 0.70,
        "topk_per_object": 1,
        "broad_support_veto": True,
    },
    {
        "variant_id": "R7AS2_anchor_support_confirm_tauS060",
        "anchor_threshold": 0.90,
        "support_threshold": 0.60,
        "topk_per_object": 1,
        "broad_support_veto": True,
    },
    {
        "variant_id": "R7AS3_anchor_support_confirm_semantic_gate",
        "anchor_threshold": 0.90,
        "support_threshold": 0.60,
        "topk_per_object": 1,
        "broad_support_veto": True,
        "semantic_gate": True,
    },
    {
        "variant_id": "R7AS4_anchor_support_confirm_veto_attenuated",
        "anchor_threshold": 0.90,
        "support_threshold": 0.55,
        "topk_per_object": 1,
        "broad_support_veto": True,
        "broad_support_min_support_mean": 1500.0,
    },
    {
        "variant_id": "R7AS5_weak_anchor_rescued_by_support_and_skeleton",
        "anchor_threshold": 0.85,
        "support_threshold": 0.60,
        "topk_per_object": 1,
        "broad_support_veto": True,
        "requires_mask_view_skeleton": True,
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
        if not np.isfinite(out):
            return default
        return out
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
        "schema_version": "stream4d_v103_r7_phase3_artifact_row_v1",
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
        "schema_version": "stream4d_v103_r7_phase3_gate_row_v1",
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
        "schema_version": "stream4d_v103_r7_phase3_failure_row_v1",
        "phase_id": PHASE_ID,
        "failure_id": failure_id,
        "severity": severity,
        "detail": json.dumps(_jsonable(detail), sort_keys=True) if isinstance(detail, (dict, list, tuple)) else detail,
        "repair_direction": repair,
        "uses_gt_for_prediction": False,
        "uses_future": False,
    }


def _feature_map(base: pd.DataFrame, object_features: dict[str, np.ndarray], scene: str) -> dict[str, np.ndarray]:
    oids = sorted(base[base["scene_id"].astype(str) == scene]["mv_object_id"].astype(str).unique().tolist())
    return {oid: object_features[oid] for oid in oids}


def _candidate_edges(
    *,
    scene: str,
    scene_base: pd.DataFrame,
    anchor_features: dict[str, np.ndarray],
    support_features: dict[str, np.ndarray],
    variant: dict[str, Any],
    shuffle_support: bool,
    rng: np.random.Generator,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    object_ids = sorted(scene_base["mv_object_id"].astype(str).unique().tolist())
    anchor_matrix = np.stack([anchor_features[oid] for oid in object_ids], axis=0)
    support_matrix = np.stack([support_features[oid] for oid in object_ids], axis=0)
    if shuffle_support and len(object_ids) > 1:
        support_matrix = support_matrix[rng.permutation(len(object_ids))]
    anchor_sim = anchor_matrix @ anchor_matrix.T
    support_sim = support_matrix @ support_matrix.T
    by_object = {oid: scene_base[scene_base["mv_object_id"].astype(str) == oid] for oid in object_ids}
    tau_a = float(variant["anchor_threshold"])
    tau_s = float(variant["support_threshold"])
    stats = {
        "A_anchor_edge_count": 0,
        "AS_confirmed_edge_count": 0,
        "AS_rejected_by_no_support_count": 0,
        "AS_rejected_by_veto_count": 0,
        "support_only_candidate_count": 0,
    }
    edges: list[dict[str, Any]] = []
    for i, oid_a in enumerate(object_ids[:-1]):
        local: list[dict[str, Any]] = []
        for j in range(i + 1, len(object_ids)):
            oid_b = object_ids[j]
            a_score = float(anchor_sim[i, j])
            s_score = float(support_sim[i, j])
            if a_score < tau_a:
                if s_score >= tau_s:
                    stats["support_only_candidate_count"] += 1
                continue
            stats["A_anchor_edge_count"] += 1
            if s_score < tau_s:
                stats["AS_rejected_by_no_support_count"] += 1
                continue
            conflict = _specific_conflict(by_object[oid_a], by_object[oid_b])
            broad = _broad_support_risk(by_object[oid_a], variant) or _broad_support_risk(by_object[oid_b], variant)
            if conflict or broad:
                stats["AS_rejected_by_veto_count"] += 1
                continue
            stats["AS_confirmed_edge_count"] += 1
            local.append(
                {
                    "object_a": oid_a,
                    "object_b": oid_b,
                    "anchor_affinity": a_score,
                    "support_affinity": s_score,
                    "combined_score": 0.5 * (a_score + s_score),
                    "specific_conflict": conflict,
                    "broad_support_veto": broad,
                    "support_only_edge_flag": False,
                    "shuffle_support": bool(shuffle_support),
                }
            )
        local.sort(key=lambda row: (row["combined_score"], row["support_affinity"], row["anchor_affinity"]), reverse=True)
        topk = int(variant.get("topk_per_object", 1))
        edges.extend(local[:topk] if topk > 0 else local)
    edges.sort(key=lambda row: (row["combined_score"], row["support_affinity"], row["anchor_affinity"]), reverse=True)
    return edges, stats


def _materialize_variant(
    *,
    base: pd.DataFrame,
    anchor_features: dict[str, np.ndarray],
    support_features: dict[str, np.ndarray],
    variant: dict[str, Any],
    chunk_id: str,
    shuffle_support: bool = False,
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    scene_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    edge_rows: list[dict[str, Any]] = []
    cluster_rows: list[dict[str, Any]] = []
    all_stats = {
        "A_anchor_edge_count": 0,
        "AS_confirmed_edge_count": 0,
        "AS_rejected_by_no_support_count": 0,
        "AS_rejected_by_veto_count": 0,
        "support_only_candidate_count": 0,
        "accepted_S_only_edge_count": 0,
    }
    rng = np.random.default_rng(7103)
    variant_id = str(variant["variant_id"]) + ("__shuffled_support_control" if shuffle_support else "")
    for scene, scene_base in base.groupby("scene_id", sort=True):
        scene = str(scene)
        object_ids = sorted(scene_base["mv_object_id"].astype(str).unique().tolist())
        uf = UnionFind(object_ids)
        edges, stats = _candidate_edges(
            scene=scene,
            scene_base=scene_base,
            anchor_features=anchor_features,
            support_features=support_features,
            variant=variant,
            shuffle_support=shuffle_support,
            rng=rng,
        )
        for key, value in stats.items():
            all_stats[key] += int(value)
        for rank, edge in enumerate(edges):
            accepted = False
            reason = ""
            if uf.find(str(edge["object_a"])) != uf.find(str(edge["object_b"])):
                uf.union(str(edge["object_a"]), str(edge["object_b"]))
                accepted = True
            edge_rows.append(
                {
                    "schema_version": "stream4d_v103_r7_phase3_edge_row_v1",
                    "phase_id": PHASE_ID,
                    "variant_id": variant_id,
                    "scene_id": scene,
                    "edge_rank": int(rank),
                    "object_a": str(edge["object_a"]),
                    "object_b": str(edge["object_b"]),
                    "anchor_affinity": float(edge["anchor_affinity"]),
                    "support_affinity": float(edge["support_affinity"]),
                    "combined_score": float(edge["combined_score"]),
                    "accepted_union": bool(accepted),
                    "reject_reason": reason,
                    "specific_conflict": bool(edge["specific_conflict"]),
                    "broad_support_veto": bool(edge["broad_support_veto"]),
                    "support_only_edge_flag": False,
                    "shuffle_support": bool(shuffle_support),
                    "uses_gt_for_prediction": False,
                    "uses_future": False,
                }
            )
        components: dict[str, list[str]] = defaultdict(list)
        for oid in object_ids:
            components[uf.find(oid)].append(oid)
        for comp_idx, members in enumerate(sorted(components.values(), key=lambda v: (-len(v), v[0]))):
            comp_rows = scene_base[scene_base["mv_object_id"].astype(str).isin(members)].copy()
            object_id = f"{variant_id}:{scene}:{chunk_id}:merged_{comp_idx:05d}"
            frames = sorted(comp_rows["frame_id"].astype(int).unique().tolist())
            base_score = float(comp_rows.get("score", pd.Series([0.0])).max())
            cluster_rows.append(
                {
                    "schema_version": "stream4d_v103_r7_phase3_cluster_row_v1",
                    "phase_id": PHASE_ID,
                    "variant_id": variant_id,
                    "scene_id": scene,
                    "mv_object_id": object_id,
                    "source_object_count": int(len(members)),
                    "frame_count": int(len(frames)),
                    "object_score": base_score,
                    "base_object_score": base_score,
                    "score_policy": "max_f2_score",
                    "selected_broad_rate": float(comp_rows["selected_mask_is_broad"].astype(bool).mean()),
                    "selected_object_like_rate": float(comp_rows["selected_mask_is_object_like"].astype(bool).mean()),
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
                        "schema_version": "stream4d_v103_r7_phase3_frame_mask_row_v1",
                        "phase_id": PHASE_ID,
                        "variant_id": variant_id,
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
                        "node_policy": "r7_anchor_confirmed_support",
                        "emit_policy": "component_wta_by_f2_score_support",
                        "readout_mode": "r7_anchor_confirmed_support",
                        "uses_gt_for_prediction": False,
                        "uses_future": False,
                    }
                )
    return scene_rows, edge_rows, cluster_rows, all_stats


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

    artifact_rows = [
        _artifact_row("r7_phase1_summary", phase1_root / "summary.json"),
        _artifact_row("f2_root_rows", f2_root / "mv_object_frame_mask_rows.parquet"),
        _artifact_row("anchor_feature_root", anchor_root, note="R6F0 phase5-like anchor-only feature root"),
        _artifact_row("support_feature_root", support_root, note="R6F2 phase5-like support-conditioned semantic feature root"),
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
    best_control_by_base: dict[str, dict[str, Any]] = {}
    for variant in VARIANTS:
        for shuffle in [False, True]:
            scene_rows, edge_rows, cluster_rows, stats = _materialize_variant(
                base=base,
                anchor_features=anchor_features,
                support_features=support_features,
                variant=variant,
                chunk_id=str(args.chunk_id),
                shuffle_support=shuffle,
            )
            variant_id = str(variant["variant_id"]) + ("__shuffled_support_control" if shuffle else "")
            window_rows, aggregate, selected_rows, pixel_collision_count, missing_count, _frame_count = _evaluate_variant(
                variant_id=variant_id,
                scene_rows=scene_rows,
                phase2_summaries=phase2_summaries,
                min_pred_pixels=int(args.min_pred_pixels),
                min_gt_pixels=int(args.min_gt_pixels),
                use_cupy_iou=not bool(args.disable_cupy_iou),
                cupy_device_id=int(args.cupy_device_id),
            )
            row = {
                "schema_version": "stream4d_v103_r7_phase3_metric_row_v1",
                "phase_id": PHASE_ID,
                "variant_id": variant_id,
                "base_variant_id": str(variant["variant_id"]),
                "variant_family": "R7AS_anchor_confirmed_support",
                "control_role": "shuffled_support_control" if shuffle else "real",
                "anchor_threshold": float(variant["anchor_threshold"]),
                "support_threshold": float(variant["support_threshold"]),
                "A_anchor_edge_count": int(stats["A_anchor_edge_count"]),
                "AS_confirmed_edge_count": int(stats["AS_confirmed_edge_count"]),
                "AS_rejected_by_no_support_count": int(stats["AS_rejected_by_no_support_count"]),
                "AS_rejected_by_veto_count": int(stats["AS_rejected_by_veto_count"]),
                "support_only_candidate_count": int(stats["support_only_candidate_count"]),
                "accepted_S_only_edge_count": int(stats["accepted_S_only_edge_count"]),
                "accepted_edge_count": int(sum(1 for r in edge_rows if bool(r["accepted_union"]))),
                "same_frame_collision_count": int(aggregate.get("same_frame_collision_count", 0)),
                "pixel_collision_rate": float(aggregate.get("pixel_collision_rate", 0.0)),
                "missing_mask_raster_count": int(missing_count),
                "MV_AP_window": float(aggregate.get("MV_AP_window", 0.0)),
                "MV_AP50_window": float(aggregate.get("MV_AP50_window", 0.0)),
                "MV_AP25_window": float(aggregate.get("MV_AP25_window", 0.0)),
                "ScoreFreeMatch50_window": float(aggregate.get("ScoreFreeMatch50_window", 0.0)),
                "uses_gt_for_prediction": False,
                "uses_gt_for_eval": True,
                "uses_future": False,
            }
            metric_rows.append(row)
            if shuffle:
                control_rows.append(row)
                best_control_by_base[str(variant["variant_id"])] = row
            elif best_real is None or float(row["MV_AP_window"]) > float(best_real["MV_AP_window"]):
                best_real = row
            variant_rows.append(
                {
                    "schema_version": "stream4d_v103_r7_phase3_variant_row_v1",
                    "phase_id": PHASE_ID,
                    "variant_id": variant_id,
                    "control_role": row["control_role"],
                    "definition": json.dumps(_jsonable(variant), sort_keys=True),
                    "uses_gt_for_prediction": False,
                    "uses_future": False,
                }
            )
            all_edge_rows.extend(edge_rows)
            all_cluster_rows.extend(cluster_rows)
            all_selected_rows.extend(selected_rows)
            all_window_rows.extend(window_rows)

    replay_mv = _num(phase1_summary.get("replay_MV_AP_window"), 0.1147308155349091)
    replay_ap50 = _num(phase1_summary.get("replay_MV_AP50_window"), 0.2871561795903901)
    for row in metric_rows:
        if row["control_role"] != "real":
            continue
        control = best_control_by_base.get(str(row["variant_id"]), {})
        if not control:
            control = best_control_by_base.get(str(row["base_variant_id"]), {})
        row["real_minus_shuffled_MV_AP_window"] = float(row["MV_AP_window"]) - _num(control.get("MV_AP_window"), 0.0)
        row["real_minus_shuffled_MV_AP50_window"] = float(row["MV_AP50_window"]) - _num(control.get("MV_AP50_window"), 0.0)
        row["real_minus_replay_MV_AP_window"] = float(row["MV_AP_window"]) - replay_mv
        row["real_minus_replay_MV_AP50_window"] = float(row["MV_AP50_window"]) - replay_ap50

    best_real = best_real or {}
    best_control = best_control_by_base.get(str(best_real.get("base_variant_id", "")), {})
    best_minus_control = _num(best_real.get("MV_AP_window"), 0.0) - _num(best_control.get("MV_AP_window"), 0.0)
    best_minus_control_ap50 = _num(best_real.get("MV_AP50_window"), 0.0) - _num(best_control.get("MV_AP50_window"), 0.0)
    subset_gate_pass = bool(
        best_real
        and _num(best_real.get("MV_AP_window"), 0.0) >= replay_mv + 0.005
        and _num(best_real.get("MV_AP50_window"), 0.0) >= replay_ap50 + 0.010
        and best_minus_control >= 0.003
        and best_minus_control_ap50 >= 0.006
        and int(best_real.get("accepted_S_only_edge_count", 1)) == 0
        and int(best_real.get("same_frame_collision_count", 1)) == 0
        and int(best_real.get("missing_mask_raster_count", 1)) == 0
        and float(best_real.get("pixel_collision_rate", 1.0)) == 0.0
    )

    gate_rows = [
        _gate_row("r7_phase1_available", bool(phase1_summary), phase1_summary.get("decision", ""), "summary readable", "Run R7-1 first."),
        _gate_row("phase1_d4rt_only", not bool(phase1_summary.get("DA3_USED", True)), phase1_summary.get("DA3_USED", ""), False, "Repair R7-1 D4RT-only boundary."),
        _gate_row("anchor_and_support_features_available", anchor_root.exists() and support_root.exists(), {"anchor": _rel(anchor_root), "support": _rel(support_root)}, "both roots", "Regenerate R6/R7 phase5-like features."),
        _gate_row("accepted_S_only_edge_count_zero", all(int(r.get("accepted_S_only_edge_count", 0)) == 0 for r in metric_rows), 0, 0, "Disallow support-only union."),
        _gate_row("best_subset_gate_pass", subset_gate_pass, best_real, "R7AS subset success thresholds", "Proceed to R7-4/R7-5; do not promote R7AS if false."),
    ]
    failure_rows: list[dict[str, Any]] = []
    if not subset_gate_pass:
        failure_rows.append(
            _failure_row(
                "NO_GO_R7_3_ANCHOR_CONFIRMED_SUPPORT_SUBSET_GATE",
                {
                    "best_real": best_real,
                    "best_control": best_control,
                    "best_real_minus_control_MV_AP_window": best_minus_control,
                    "best_real_minus_control_MV_AP50_window": best_minus_control_ap50,
                    "replay_MV_AP_window": replay_mv,
                    "replay_MV_AP50_window": replay_ap50,
                },
                "Try R7-4 skeleton-confirmed support and R7-5 anchor-seeded support propagation before any full-dev promotion.",
            )
        )
    else:
        failure_rows.append(_failure_row("NONE", "R7-3 subset gate passed.", "Freeze config before full-dev.", severity="info"))

    summary = {
        "schema_version": "stream4d_v103_r7_phase3_summary_v1",
        "phase": "R7-3",
        "phase_id": PHASE_ID,
        "phase_pass": subset_gate_pass,
        "decision": "PASS_R7_3_SUBSET_CANDIDATE" if subset_gate_pass else "NO_GO_R7_3_ANCHOR_CONFIRMED_SUPPORT",
        "runtime_sec": time.time() - t0,
        "variant_count": len(VARIANTS),
        "metric_row_count": len(metric_rows),
        "edge_row_count": len(all_edge_rows),
        "best_real_variant_id": best_real.get("variant_id", ""),
        "best_real_MV_AP_window": best_real.get("MV_AP_window", ""),
        "best_real_MV_AP50_window": best_real.get("MV_AP50_window", ""),
        "best_real_minus_replay_MV_AP_window": _num(best_real.get("MV_AP_window"), 0.0) - replay_mv if best_real else "",
        "best_real_minus_replay_MV_AP50_window": _num(best_real.get("MV_AP50_window"), 0.0) - replay_ap50 if best_real else "",
        "best_real_minus_shuffled_MV_AP_window": best_minus_control,
        "best_real_minus_shuffled_MV_AP50_window": best_minus_control_ap50,
        "accepted_S_only_edge_count": int(best_real.get("accepted_S_only_edge_count", 0)) if best_real else "",
        "same_frame_collision_count": int(best_real.get("same_frame_collision_count", 0)) if best_real else "",
        "pixel_collision_rate": float(best_real.get("pixel_collision_rate", 0.0)) if best_real else "",
        "missing_mask_raster_count": int(best_real.get("missing_mask_raster_count", 0)) if best_real else "",
        "uses_gt_for_prediction": False,
        "uses_gt_for_eval": True,
        "uses_future": False,
        "truthfulness_note": "R7-3 requires both anchor affinity and support affinity for every edge; support-only union is never materialized. GT is used only by the evaluator.",
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
    }

    _write_csv(out / "artifact_rows.csv", artifact_rows)
    _write_csv(out / "gate_rows.csv", gate_rows)
    _write_csv(out / "failure_rows.csv", failure_rows)
    _write_csv(out / "variant_rows.csv", variant_rows)
    _write_csv(out / "metric_rows.csv", metric_rows)
    _write_csv(out / "control_rows.csv", control_rows)
    _write_csv(out / "edge_rows.csv", all_edge_rows)
    _write_csv(out / "cluster_rows.csv", all_cluster_rows)
    _write_csv(out / "selected_rows.csv", all_selected_rows)
    _write_csv(out / "window_rows.csv", all_window_rows)
    _write_json(out / "summary.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run R7-3 anchor-confirmed support variants.")
    parser.add_argument("--output-root", default=str(DEFAULT_OUT))
    parser.add_argument("--r7-phase1-root", default=str(DEFAULT_R7_PHASE1_ROOT))
    parser.add_argument("--f2-root", default=str(DEFAULT_F2_ROOT))
    parser.add_argument("--anchor-feature-root", default=str(DEFAULT_ANCHOR_FEATURE))
    parser.add_argument("--support-feature-root", default=str(DEFAULT_SUPPORT_FEATURE))
    parser.add_argument("--scene0011-phase2-root", default=str(DEFAULT_SCENE0011_PHASE2))
    parser.add_argument("--scene0050-phase2-root", default=str(DEFAULT_SCENE0050_PHASE2))
    parser.add_argument("--dataset-split", default="dev")
    parser.add_argument("--chunk-id", default="c0001")
    parser.add_argument("--min-pred-pixels", type=int, default=20)
    parser.add_argument("--min-gt-pixels", type=int, default=20)
    parser.add_argument("--cupy-device-id", type=int, default=0)
    parser.add_argument("--disable-cupy-iou", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> int:
    summary = build(build_parser().parse_args())
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
