#!/usr/bin/env python3
"""Audit whether current artifacts can build object-specific v94 field tensors."""

from __future__ import annotations

import csv
import hashlib
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PHASE_ID = "v94_phase7b_object_specific_field_readiness"
RUN_ID = "v94_phase7b_object_specific_field_readiness"
OUT = ROOT / "outputs/audit/v94_phase7b_object_specific_field_readiness"

PHASE1 = ROOT / "outputs/audit/v94_phase1_canonical_graph"
V93_REGION = ROOT / "outputs/audit/v93_phase3_region_edge_graph"
V91_MASK_FEATURE_STORE = ROOT / "outputs/audit/v91_radio_mask_features_npz"
V91_PROTO_QUALITY = ROOT / "outputs/audit/v91_radio_feature_store_quality"


def _created_at() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _jsonable(row.get(key, "")) for key in fieldnames})


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def _csv_header(path: Path) -> list[str]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        return next(reader, [])


def _object_stats(path: Path) -> dict[str, Any]:
    rows = 0
    nonempty = Counter()
    semantic_proto_values: set[str] = set()
    local_cluster_values: set[str] = set()
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows += 1
            for key, value in row.items():
                if value not in (None, ""):
                    nonempty[key] += 1
            if row.get("semantic_proto"):
                semantic_proto_values.add(str(row["semantic_proto"]))
            if row.get("local_cluster_id"):
                local_cluster_values.add(str(row["local_cluster_id"]))
    return {
        "object_hypothesis_rows": rows,
        "local_cluster_nonempty_count": nonempty["local_cluster_id"],
        "local_cluster_unique_count": len(local_cluster_values),
        "semantic_proto_nonempty_count": nonempty["semantic_proto"],
        "semantic_proto_unique_count": len(semantic_proto_values),
        "appearance_feature_hash_nonempty_count": nonempty["appearance_feature_hash"],
        "object_score_nonempty_count": nonempty["object_score"],
        "hard_negative_density_nonempty_count": nonempty["hard_negative_density"],
    }


def _v91_proto_stats(path: Path) -> dict[str, Any]:
    rows = 0
    diagnostic_gt_rows = 0
    local_slots: set[str] = set()
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows += 1
            if str(row.get("diagnostic_only_uses_gt", "")).lower() in {"1", "true", "yes"}:
                diagnostic_gt_rows += 1
            if row.get("local_slot_id"):
                local_slots.add(str(row["local_slot_id"]))
    return {
        "v91_slot_proto_rows": rows,
        "v91_slot_proto_local_slot_count": len(local_slots),
        "v91_slot_proto_diagnostic_gt_rows": diagnostic_gt_rows,
        "v91_slot_proto_diagnostic_gt_rate": float(diagnostic_gt_rows / rows) if rows else 0.0,
    }


def _mask_feature_store_stats(root: Path) -> dict[str, Any]:
    npz_path = root / "mask_features.npz"
    index_path = root / "mask_feature_index.csv"
    stats: dict[str, Any] = {
        "v91_mask_feature_store_exists": npz_path.exists(),
        "v91_mask_feature_index_exists": index_path.exists(),
    }
    if npz_path.exists():
        with np.load(npz_path, allow_pickle=False) as data:
            features = data["features"]
            stats.update(
                {
                    "v91_mask_feature_vector_count": int(features.shape[0]),
                    "v91_mask_feature_dim": int(features.shape[1]) if features.ndim == 2 else "",
                    "v91_mask_feature_backend": str(data["backend"].item()) if "backend" in data and data["backend"].shape == () else "",
                    "v91_mask_feature_layer": str(data["layer"].item()) if "layer" in data and data["layer"].shape == () else "",
                }
            )
    return stats


def run() -> dict[str, Any]:
    OUT.mkdir(parents=True, exist_ok=True)
    created_at = _created_at()

    object_path = PHASE1 / "object_hypothesis_rows.csv"
    region_feature_path = V93_REGION / "region_feature_rows.csv"
    region_node_path = V93_REGION / "region_node_rows.csv"
    v91_proto_path = V91_PROTO_QUALITY / "radio_slot_proto_rows.csv"

    object_stats = _object_stats(object_path)
    region_feature_header = _csv_header(region_feature_path)
    region_node_header = _csv_header(region_node_path)
    v91_proto_stats = _v91_proto_stats(v91_proto_path) if v91_proto_path.exists() else {}
    mask_store_stats = _mask_feature_store_stats(V91_MASK_FEATURE_STORE)

    region_has_vector_columns = any(name.startswith("radio_feature_") and name not in {"radio_feature_ref", "radio_feature_norm"} for name in region_feature_header)
    region_has_only_refs = "radio_feature_ref" in region_feature_header and not region_has_vector_columns
    object_has_appearance_vectors = object_stats["appearance_feature_hash_nonempty_count"] > 0
    method_safe_v91_proto = bool(v91_proto_stats) and v91_proto_stats.get("v91_slot_proto_diagnostic_gt_rate", 1.0) == 0.0
    mask_store_can_be_direct_object_axis = bool(mask_store_stats.get("v91_mask_feature_store_exists")) and object_has_appearance_vectors

    evidence_rows = [
        {
            "schema_version": "stream4d_v94_phase7b_evidence_source_v1",
            "phase_id": PHASE_ID,
            "run_id": RUN_ID,
            "source_name": "v94_object_hypothesis_rows",
            "artifact": _rel(object_path),
            "available": object_path.exists(),
            "method_safe": True,
            "object_specific": True,
            "vector_available": object_has_appearance_vectors,
            "mapped_to_v94_objects": True,
            "notes": "appearance_feature_hash is the only object-vector pointer in this schema; it is empty in current artifacts.",
        },
        {
            "schema_version": "stream4d_v94_phase7b_evidence_source_v1",
            "phase_id": PHASE_ID,
            "run_id": RUN_ID,
            "source_name": "v93_region_feature_rows",
            "artifact": _rel(region_feature_path),
            "available": region_feature_path.exists(),
            "method_safe": True,
            "object_specific": False,
            "vector_available": region_has_vector_columns,
            "mapped_to_v94_objects": False,
            "notes": "Current rows expose region radio_feature_ref/norm, not persisted region vectors for object-prototype cosine.",
        },
        {
            "schema_version": "stream4d_v94_phase7b_evidence_source_v1",
            "phase_id": PHASE_ID,
            "run_id": RUN_ID,
            "source_name": "v91_mask_feature_store",
            "artifact": _rel(V91_MASK_FEATURE_STORE / "mask_features.npz"),
            "available": bool(mask_store_stats.get("v91_mask_feature_store_exists", False)),
            "method_safe": True,
            "object_specific": False,
            "vector_available": bool(mask_store_stats.get("v91_mask_feature_store_exists", False)),
            "mapped_to_v94_objects": mask_store_can_be_direct_object_axis,
            "notes": "Mask-level vectors exist, but current v94 object rows do not carry appearance hashes/indices that make them direct object-axis prototypes.",
        },
        {
            "schema_version": "stream4d_v94_phase7b_evidence_source_v1",
            "phase_id": PHASE_ID,
            "run_id": RUN_ID,
            "source_name": "v91_radio_slot_proto_quality",
            "artifact": _rel(v91_proto_path),
            "available": v91_proto_path.exists(),
            "method_safe": method_safe_v91_proto,
            "object_specific": True,
            "vector_available": False,
            "mapped_to_v94_objects": False,
            "notes": "Rows are diagnostic-quality cosine scores with diagnostic_only_uses_gt=true, not a method-safe prototype vector table.",
        },
    ]

    gate_pass = bool(object_has_appearance_vectors and region_has_vector_columns)
    if gate_pass:
        decision = "PASS_V94_OBJECT_SPECIFIC_FIELD_INPUTS_AVAILABLE"
        blocker = ""
        repair = "Build GPU object-axis field shards from persisted object and region vectors."
    else:
        decision = "NO_GO_V94_OBJECT_SPECIFIC_FIELD_INPUTS_NOT_AVAILABLE"
        blocker = "missing_method_safe_object_region_vector_pair_for_object_specific_unary"
        repair = "Persist method-safe region vectors and canonical object prototype vectors keyed by v94 canonical object id, then rebuild GPU field shards with source x object x region unary tensors."

    summary = {
        "schema": "stream4d_v94_phase7b_object_specific_field_readiness_summary_v1",
        "phase_id": PHASE_ID,
        "run_id": RUN_ID,
        "created_at": created_at,
        "decision": decision,
        "object_specific_field_input_gate_pass": gate_pass,
        "object_specific_field_blocker": blocker,
        "recommended_repair_direction": repair,
        **object_stats,
        **mask_store_stats,
        **v91_proto_stats,
        "region_feature_header_has_radio_feature_ref": "radio_feature_ref" in region_feature_header,
        "region_feature_header_has_region_vectors": region_has_vector_columns,
        "region_feature_header_mode": "feature_ref_only" if region_has_only_refs else "vector_columns_or_missing",
        "region_node_header_has_source_mean_cosine": "source_mean_cosine" in region_node_header,
        "uses_gt_for_prediction": False,
        "uses_future": False,
        "row_counts": {"evidence_source_rows": len(evidence_rows)},
        "input_artifacts": {
            _rel(path): _sha256(path)
            for path in [
                object_path,
                region_feature_path,
                region_node_path,
                V91_MASK_FEATURE_STORE / "mask_features.npz",
                V91_MASK_FEATURE_STORE / "mask_feature_index.csv",
                v91_proto_path,
            ]
            if path.exists()
        },
    }
    _write_csv(OUT / "evidence_source_rows.csv", evidence_rows)
    _write_json(OUT / "summary.json", summary)
    outputs = [OUT / "evidence_source_rows.csv", OUT / "summary.json"]
    _write_json(OUT / "SHA256SUMS.json", {_rel(path): _sha256(path) for path in outputs if path.exists()})
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True), flush=True)
    return summary


if __name__ == "__main__":
    run()
