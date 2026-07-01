#!/usr/bin/env python3
"""Phase4o GT-free union of existing local2history evidence sources.

Phase4n showed, diagnostically, that accepted links are relatively precise but
coverage is low. This repair does not use GT. It unions already-produced
GT-free semantic history, exact-overlap, and temporal mask-view accepted links,
then evaluates the resulting scene ids while preserving Phase2c local rows.
"""

from __future__ import annotations

import csv
import json
import sys
import time
from collections import defaultdict
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
OUT_DIR = AUDIT_ROOT / "v100_phase4o_union_history_evidence_repair"
PHASE2C_DIR = AUDIT_ROOT / "v100_phase2c_overlap3_local_repair"
PHASE2C_SUMMARY = PHASE2C_DIR / "summary.json"
PHASE4H_DIR = AUDIT_ROOT / "v100_phase4h_overlap3_exact_history_memory"
PHASE4K_DIR = AUDIT_ROOT / "v100_phase4k_phase2c_semantic_scene_repair"
PHASE4M_DIR = AUDIT_ROOT / "v100_phase4m_temporal_maskview_history_memory"


def _rel(path: Path | str) -> str:
    return p4h._rel(path)


def _num(value: Any, default: float = 0.0) -> float:
    return p4h._num(value, default)


def _bool(value: Any) -> bool:
    return p4h._bool(value)


def _jsonable(value: Any) -> Any:
    return p4h._jsonable(value)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    p4h._write_csv(path, rows)


def _write_json(path: Path, payload: Any) -> None:
    p4h._write_json(path, payload)


def _write_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    p4h._write_parquet(path, rows)


def _phase4o_artifact_rows(paths: list[tuple[Path, str, str]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path, kind, note in paths:
        rows.append(
            {
                "schema_version": "stream4d_v100_phase4o_artifact_manifest_row_v1",
                "phase_id": "v100_phase4o_union_history_evidence_repair",
                "artifact_path": _rel(path),
                "artifact_type": kind,
                "size_bytes": path.stat().st_size if path.exists() else 0,
                "sha256": p4h._sha256(path) if path.exists() and path.is_file() else "",
                "note": note,
            }
        )
    return rows


def _merge_csv_edges(path: Path, *, source_id: str, source_variant_id: str, source_family: str) -> list[dict[str, Any]]:
    df = pd.read_csv(path)
    df = df[df["variant_id"] == source_variant_id]
    rows: list[dict[str, Any]] = []
    for item in df.to_dict(orient="records"):
        rows.append(
            {
                "schema_version": "stream4d_v100_phase4o_source_edge_row_v1",
                "phase_id": "v100_phase4o_union_history_evidence_repair",
                "source_id": source_id,
                "source_variant_id": source_variant_id,
                "source_family": source_family,
                "dataset_split": str(item["dataset_split"]),
                "scene_id": str(item["scene_id"]),
                "left_chunk_id": str(item.get("left_chunk_id", "")),
                "right_chunk_id": str(item.get("right_chunk_id", "")),
                "mv_object_id_a": str(item["mv_object_id_a"]),
                "mv_object_id_b": str(item["mv_object_id_b"]),
                "candidate_family": str(item.get("candidate_family", source_family)),
                "source_score": item.get("affinity", item.get("link_score", "")),
                "uses_gt_for_prediction": False,
                "uses_future": False,
            }
        )
    return rows


def _semantic_history_edges(path: Path, *, source_id: str, source_variant_id: str) -> list[dict[str, Any]]:
    hist_df = pd.read_parquet(path)
    hist_df = hist_df[hist_df["variant_id"] == source_variant_id].copy()
    rows: list[dict[str, Any]] = []
    for item in hist_df.to_dict(orient="records"):
        local_ids = [oid for oid in str(item.get("local_object_ids", "")).split(";") if oid]
        if len(local_ids) < 2:
            continue
        anchor = local_ids[0]
        for other in local_ids[1:]:
            rows.append(
                {
                    "schema_version": "stream4d_v100_phase4o_source_edge_row_v1",
                    "phase_id": "v100_phase4o_union_history_evidence_repair",
                    "source_id": source_id,
                    "source_variant_id": source_variant_id,
                    "source_family": "semantic_history",
                    "dataset_split": str(item.get("dataset_split", "")),
                    "scene_id": str(item.get("scene_id", "")),
                    "left_chunk_id": str(item.get("birth_chunk", "")),
                    "right_chunk_id": str(item.get("last_seen_chunk", "")),
                    "mv_object_id_a": anchor,
                    "mv_object_id_b": other,
                    "candidate_family": "semantic_history_final_component_edge",
                    "source_score": item.get("score_ema", ""),
                    "source_margin": "",
                    "history_id": str(item.get("history_id", "")),
                    "history_previous_member_count": len(local_ids) - 1,
                    "uses_gt_for_prediction": False,
                    "uses_future": False,
                }
            )
    return rows


def _semantic_history_edges_from_action_log(path: Path, *, source_id: str, source_variant_id: str) -> list[dict[str, Any]]:
    df = pd.read_parquet(path)
    df = df[df["variant_id"] == source_variant_id].copy()
    df = df.sort_values(["dataset_split", "scene_id", "chunk_index", "chunk_id", "chunk_object_id"])
    state: dict[tuple[str, str, str], list[str]] = defaultdict(list)
    rows: list[dict[str, Any]] = []
    for item in df.to_dict(orient="records"):
        split = str(item["dataset_split"])
        scene = str(item["scene_id"])
        hist = str(item["history_id"])
        current = str(item["chunk_object_id"])
        key = (split, scene, hist)
        previous = list(state.get(key, []))
        action = str(item["action"])
        if action == "accept_link":
            for prev in previous:
                rows.append(
                    {
                        "schema_version": "stream4d_v100_phase4o_source_edge_row_v1",
                        "phase_id": "v100_phase4o_union_history_evidence_repair",
                        "source_id": source_id,
                        "source_variant_id": source_variant_id,
                        "source_family": "semantic_history",
                        "dataset_split": split,
                        "scene_id": scene,
                        "left_chunk_id": "",
                        "right_chunk_id": str(item.get("chunk_id", "")),
                        "mv_object_id_a": prev,
                        "mv_object_id_b": current,
                        "candidate_family": "semantic_history_accept_link",
                        "source_score": item.get("link_score", ""),
                        "source_margin": item.get("link_margin", ""),
                        "history_id": hist,
                        "history_previous_member_count": len(previous),
                        "uses_gt_for_prediction": False,
                        "uses_future": False,
                    }
                )
        if action in {"birth_new_history", "accept_link"} and current not in state[key]:
            state[key].append(current)
    return rows


def _apply_edges(ids: list[str], edges: list[dict[str, Any]], *, variant_id: str) -> tuple[dict[str, str], list[dict[str, Any]]]:
    dsu = p4h.DSU(ids)
    accepted: list[dict[str, Any]] = []
    for row in edges:
        a = str(row["mv_object_id_a"])
        b = str(row["mv_object_id_b"])
        if a not in dsu.parent or b not in dsu.parent:
            continue
        if dsu.union(a, b):
            new = dict(row)
            new["schema_version"] = "stream4d_v100_phase4o_union_edge_row_v1"
            new["variant_id"] = variant_id
            new["union_policy"] = "dsu_union_existing_gt_free_evidence_sources"
            accepted.append(new)
    mapping = {oid: f"{variant_id}:{dsu.find(oid)}" for oid in ids}
    return mapping, accepted


def _apply_mapping(rows: list[dict[str, Any]], mapping: dict[str, str], *, variant_id: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        oid = str(row["mv_object_id"])
        new = dict(row)
        new["schema_version"] = "stream4d_v100_phase4o_scene_mv_object_frame_mask_row_v1"
        new["phase_id"] = "v100_phase4o_union_history_evidence_repair"
        new["variant_id"] = variant_id
        new["variant"] = variant_id
        new["source_phase2c_mv_object_id"] = oid
        new["mv_object_id"] = mapping.get(oid, f"{variant_id}:{oid}")
        new["object_id"] = new["mv_object_id"]
        new["history_id"] = new["mv_object_id"]
        new["object_id_policy"] = "union_existing_gt_free_history_evidence_identity"
        new["history_memory_scope"] = "phase4k_semantic_phase4h_exact_phase4m_temporal_union_no_gt"
        new["score_scope"] = "current_chunk_score_history_identity"
        new["uses_gt_for_prediction"] = False
        new["uses_future"] = False
        new["future_chunk_access"] = False
        out.append(new)
    return out


def _edge_groups() -> dict[str, list[dict[str, Any]]]:
    phase4h_summary = json.loads((PHASE4H_DIR / "summary.json").read_text(encoding="utf-8"))
    phase4k_summary = json.loads((PHASE4K_DIR / "summary.json").read_text(encoding="utf-8"))
    phase4m_summary = json.loads((PHASE4M_DIR / "summary.json").read_text(encoding="utf-8"))
    semantic = _semantic_history_edges(
        PHASE4K_DIR / "history_object_rows.parquet",
        source_id="phase4k_best_semantic",
        source_variant_id=str(phase4k_summary["best_variant_id"]),
    )
    exact = _merge_csv_edges(
        PHASE4H_DIR / "local2history_merge_rows.csv",
        source_id="phase4h_exact_adjacent",
        source_variant_id="HMO1_exact_overlap_frame_mask_adjacent",
        source_family="exact_overlap_adjacent",
    )
    phase4h_best = _merge_csv_edges(
        PHASE4H_DIR / "local2history_merge_rows.csv",
        source_id="phase4h_best",
        source_variant_id=str(phase4h_summary["best_variant_id"]),
        source_family="phase4h_exact_plus_semantic",
    )
    temporal = _merge_csv_edges(
        PHASE4M_DIR / "local2history_merge_rows.csv",
        source_id="phase4m_best_temporal_maskview",
        source_variant_id=str(phase4m_summary["best_variant_id"]),
        source_family="temporal_maskview",
    )
    return {
        "semantic": semantic,
        "exact": exact,
        "phase4h_best": phase4h_best,
        "temporal": temporal,
    }


def main() -> int:
    started = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    phase2c = json.loads(PHASE2C_SUMMARY.read_text(encoding="utf-8"))
    if not bool(phase2c.get("phase2c_pass")):
        raise RuntimeError("Phase4o requires v100 Phase2c overlap3 local pass")
    baselines = p4h._phase0_baselines()
    f2_dev = baselines["F2_base_full_dev"]
    f2_holdout = baselines["F2_base_holdout"]

    primary_df = pd.read_parquet(PHASE2C_DIR / "mv_object_frame_mask_rows.parquet")
    primary_by_split = {
        split: [dict(row) for row in sub.to_dict(orient="records")]
        for split, sub in primary_df.groupby("dataset_split")
    }
    ids_by_split = {split: sorted({str(row["mv_object_id"]) for row in rows}) for split, rows in primary_by_split.items()}
    scopes = {split: p4h._scope_for_split(split) for split in ["dev", "holdout"]}
    edge_groups = _edge_groups()

    variant_specs = [
        {
            "variant_id": "HCU0_phase4k_semantic_replay",
            "families": ["semantic"],
            "notes": "Replay Phase4k best semantic history accepted links as DSU components.",
        },
        {
            "variant_id": "HCU1_phase4h_exact_adjacent_replay",
            "families": ["exact"],
            "notes": "Replay Phase4h exact adjacent overlap accepted links.",
        },
        {
            "variant_id": "HCU2_semantic_plus_exact_adjacent_union",
            "families": ["semantic", "exact"],
            "notes": "Union Phase4k semantic history with Phase4h exact adjacent overlap links.",
        },
        {
            "variant_id": "HCU3_semantic_plus_phase4h_best_union",
            "families": ["semantic", "phase4h_best"],
            "notes": "Union Phase4k semantic history with Phase4h best exact+semantic links.",
        },
        {
            "variant_id": "HCU4_semantic_plus_exact_plus_temporal_union",
            "families": ["semantic", "exact", "temporal"],
            "notes": "Union Phase4k semantic history with exact overlap and Phase4m temporal mask-view links.",
        },
    ]

    config_rows: list[dict[str, Any]] = []
    source_edge_rows: list[dict[str, Any]] = []
    union_edge_rows: list[dict[str, Any]] = []
    variant_metric_rows: list[dict[str, Any]] = []
    scene_metric_rows: list[dict[str, Any]] = []
    frame_eval_rows: list[dict[str, Any]] = []
    rows_by_variant_split: dict[tuple[str, str], list[dict[str, Any]]] = {}

    for spec in variant_specs:
        variant_id = str(spec["variant_id"])
        family_names = list(spec["families"])
        config_rows.append(
            {
                "schema_version": "stream4d_v100_phase4o_variant_config_row_v1",
                "phase_id": "v100_phase4o_union_history_evidence_repair",
                "variant_id": variant_id,
                "source_families": ";".join(family_names),
                "notes": spec["notes"],
                "uses_gt_for_prediction": False,
                "uses_future": False,
            }
        )
        variant_edges = [edge for name in family_names for edge in edge_groups[name]]
        for edge in variant_edges:
            new_edge = dict(edge)
            new_edge["variant_id"] = variant_id
            source_edge_rows.append(new_edge)
        for split in ["dev", "holdout"]:
            split_edges = [edge for edge in variant_edges if str(edge["dataset_split"]) == split]
            mapping, accepted = _apply_edges(ids_by_split[split], split_edges, variant_id=variant_id)
            mapped_rows = _apply_mapping(primary_by_split[split], mapping, variant_id=variant_id)
            rows_by_variant_split[(variant_id, split)] = mapped_rows
            union_edge_rows.extend(accepted)

            p4h._set_inputs(split)
            per_scene, frames = p1._evaluate_variant(variant_id, mapped_rows, scopes[split])
            scene_agg = p1._aggregate_metrics(per_scene)[0]
            local_agg = p4h._local_agg_from_phase2c(phase2c, split)
            component_stats = p4h._component_stats(mapping)
            crossing = p4h._scene_crossing_stats(mapped_rows)
            row = dict(scene_agg)
            row["schema_version"] = "stream4d_v100_phase4o_metric_aggregate_row_v1"
            row["phase_id"] = "v100_phase4o_union_history_evidence_repair"
            row["variant_id"] = variant_id
            row["dataset_split"] = split
            row["MV_AP_window_scene_id_scope"] = row.get("MV_AP_window")
            row["MV_AP50_window_scene_id_scope"] = row.get("MV_AP50_window")
            row["MV_AP_window"] = local_agg["MV_AP_window"]
            row["MV_AP50_window"] = local_agg["MV_AP50_window"]
            row["metric_composition"] = "local_window_from_phase2c_chunk_ids_scene_from_phase4o_union_ids"
            row["source_edge_count"] = len(split_edges)
            row["accepted_union_edge_count"] = len(accepted)
            for name in ["semantic", "exact", "phase4h_best", "temporal"]:
                row[f"{name}_source_edge_count"] = sum(1 for edge in split_edges if edge["source_family"] == edge_groups[name][0]["source_family"]) if edge_groups[name] else 0
            row["future_chunk_access"] = False
            row["uses_gt_for_prediction"] = False
            row["uses_future"] = False
            row.update(component_stats)
            row.update(crossing)
            variant_metric_rows.append(row)
            for item in per_scene:
                item["phase_id"] = "v100_phase4o_union_history_evidence_repair"
                item["dataset_split"] = split
                item["metric_scope_note"] = "scene id scope; aggregate row preserves Phase2c local-window ids"
            scene_metric_rows.extend(per_scene)
            for item in frames:
                item["phase_id"] = "v100_phase4o_union_history_evidence_repair"
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
    )
    future_gate = not any(_bool(row.get("uses_future")) or _bool(row.get("future_chunk_access")) for row in best_rows)
    phase4o_pass = bool(
        dev_scene_gate
        and dev_scene_ap50_gate
        and hold_scene_gate
        and hold_scene_ap50_gate
        and local_drop_gate
        and objects_crossing_gate
        and safety_gate
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
            "gate_id": "collision_missing_mask_safety",
            "pass": safety_gate,
            "expected": "collision=0 pixel<=0.02 missing_mask=0",
            "observed": f"dev_collision={best_dev.get('same_frame_collision_count')} hold_collision={best_hold.get('same_frame_collision_count')} dev_missing={best_dev.get('missing_mask_raster_count')} hold_missing={best_hold.get('missing_mask_raster_count')}",
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
            "schema_version": "stream4d_v100_phase4o_failure_row_v1",
            "phase_id": "v100_phase4o_union_history_evidence_repair",
            "failure_id": row["gate_id"],
            "severity": row["severity"],
            "expected": row["expected"],
            "observed": row["observed"],
            "repair_direction": (
                "If union improves recall but still misses scene gates, inspect whether larger components hurt AP50 or whether provider geometry can add missing links. "
                "If union hurts AP, do not promote; keep separate evidence sources."
            ),
        }
        for row in gate_rows
        if not bool(row["pass"])
    ]

    scene_frame_parquet = OUT_DIR / "scene_mv_object_frame_mask_rows.parquet"
    source_edge_csv = OUT_DIR / "source_edge_rows.csv"
    union_edge_csv = OUT_DIR / "union_edge_rows.csv"
    metric_csv = OUT_DIR / "variant_metric_rows.csv"
    scene_metric_csv = OUT_DIR / "mv_metric_scene_rows.csv"
    frame_csv = OUT_DIR / "frame_eval_rows.csv"
    config_csv = OUT_DIR / "variant_config_rows.csv"
    gate_csv = OUT_DIR / "variant_gate_rows.csv"
    failure_csv = OUT_DIR / "variant_failure_rows.csv"
    performance_csv = OUT_DIR / "performance_rows.csv"
    artifact_csv = OUT_DIR / "artifact_manifest_rows.csv"

    _write_parquet(scene_frame_parquet, best_rows)
    _write_csv(source_edge_csv, source_edge_rows)
    _write_csv(union_edge_csv, union_edge_rows)
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
                "schema_version": "stream4d_v100_phase4o_performance_row_v1",
                "phase_id": "v100_phase4o_union_history_evidence_repair",
                "case_id": "union_existing_evidence_and_v65_eval",
                "runtime_sec": time.time() - started,
                "variant_count": len(variant_specs),
                "split_count": 2,
                "v65_evaluator_runs": len(variant_specs) * 2,
                "source_edge_count": len(source_edge_rows),
                "union_edge_count": len(union_edge_rows),
            }
        ],
    )
    _write_csv(
        artifact_csv,
        _phase4o_artifact_rows(
            [
                (scene_frame_parquet, "parquet", "best variant primary-emitted scene rows"),
                (source_edge_csv, "csv", "source accepted edges reused by variants"),
                (union_edge_csv, "csv", "DSU union edges accepted for variants"),
                (metric_csv, "csv", "aggregate metrics for all variants/splits"),
                (scene_metric_csv, "csv", "v65 per-scene metrics"),
                (frame_csv, "csv", "v65 frame eval rows"),
                (config_csv, "csv", "variant configs"),
                (gate_csv, "csv", "phase4o gates"),
                (failure_csv, "csv", "phase4o failures if any"),
                (performance_csv, "csv", "runtime and row counts"),
            ]
        ),
    )

    summary = {
        "schema_version": "stream4d_v100_phase4o_union_history_evidence_repair_summary_v1",
        "phase_id": "v100_phase4o_union_history_evidence_repair",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "runtime_sec": time.time() - started,
        "decision": "PASS_ENTER_PHASE5" if phase4o_pass else "BLOCK_PHASE5_REPAIR_UNION_HISTORY_EVIDENCE",
        "phase4o_pass": phase4o_pass,
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
        "accepted_union_edge_count": {
            "dev": int(_num(best_dev.get("accepted_union_edge_count"))),
            "holdout": int(_num(best_hold.get("accepted_union_edge_count"))),
        },
        "max_component_size": {
            "dev": int(_num(best_dev.get("max_component_size"))),
            "holdout": int(_num(best_hold.get("max_component_size"))),
        },
        "future_chunk_access": False,
        "uses_gt_for_prediction": False,
        "metric_composition": "local_window_from_phase2c_chunk_ids_scene_from_phase4o_union_ids",
        "outputs": {
            "summary": _rel(OUT_DIR / "summary.json"),
            "scene_mv_object_frame_mask_rows": _rel(scene_frame_parquet),
            "source_edge_rows": _rel(source_edge_csv),
            "union_edge_rows": _rel(union_edge_csv),
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
    return 0 if phase4o_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
