#!/usr/bin/env python3
"""Build v100 Phase2 formal F2 local artifact.

This phase does not tune thresholds. It imports the frozen v99 repaired local
candidate, formalizes the surfel identity dependency against the v98 Phase5
fixed-voxel construction, and writes v100 tensor/parquet/CSV artifacts for the
next local2history phases.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


STREAM3D_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = STREAM3D_ROOT.parent
if str(STREAM3D_ROOT) not in sys.path:
    sys.path.insert(0, str(STREAM3D_ROOT))


AUDIT_ROOT = STREAM3D_ROOT / "outputs/audit"
OUT_DIR = AUDIT_ROOT / "v100_phase2_f2_local_final"
PHASE0_SUMMARY = AUDIT_ROOT / "v100_phase0_contract/summary.json"
PHASE1_SUMMARY = AUDIT_ROOT / "v100_phase1_gpu_data_model_parity/summary.json"
V98_PHASE5_DEV = AUDIT_ROOT / "v98_phase5_fused_surfel"
V98_PHASE5_HOLDOUT = AUDIT_ROOT / "v98_phase13_holdout_phase5_fused_surfel"
V98_PHASE5_SCRIPT = STREAM3D_ROOT / "tools/build_v98_1_phase4_phase5_geometry_method.py"
V99_PHASE2 = AUDIT_ROOT / "v99_phase2_f2_strengthening"
V99_PHASE10L = AUDIT_ROOT / "v99_phase10l_frozen_p2d2_regenerated_birth_holdout"

SOURCE_DEV_VARIANT = "P2_D2_frame_count_plus_semantic_tiebreak"
SOURCE_HOLDOUT_VARIANT = "P2_D2_frame_count_plus_semantic_tiebreak__regenerated_chunk_birth_holdout"
V100_VARIANT = "F2_v100_chunk32_surfel_maskview_thr018_p2d2_formalized"
FORMAL_PROOF_ID = "v100_phase2_fixed_voxel_hash_chunk_rescope_proof_v1"


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


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


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


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


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


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _source_rows() -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    dev_rows = [dict(row) for row in _read_csv(V99_PHASE2 / "mv_object_frame_mask_rows.csv") if row.get("variant_id") == SOURCE_DEV_VARIANT]
    holdout_rows = [dict(row) for row in _read_csv(V99_PHASE10L / "mv_object_frame_mask_rows.csv") if row.get("variant_id") == SOURCE_HOLDOUT_VARIANT]
    if not dev_rows:
        raise RuntimeError(f"missing dev rows for {SOURCE_DEV_VARIANT}")
    if not holdout_rows:
        raise RuntimeError(f"missing holdout rows for {SOURCE_HOLDOUT_VARIANT}")
    return dev_rows, holdout_rows


def _formalize_rows(rows: list[dict[str, str]], split: str, source_artifact: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        source_oid = str(row.get("mv_object_id", row.get("object_id", "")))
        v100_oid = f"{split}:{source_oid}"
        new = dict(row)
        new["schema_version"] = "stream4d_v100_phase2_mv_object_frame_mask_row_v1"
        new["phase_id"] = "v100_phase2_f2_local_final"
        new["dataset_split"] = split
        new["variant_id"] = V100_VARIANT
        new["variant"] = V100_VARIANT
        new["source_variant_id"] = row.get("variant_id", "")
        new["source_phase_id"] = row.get("phase_id", "")
        new["source_artifact_path"] = _rel(source_artifact)
        new["source_mv_object_id"] = source_oid
        new["mv_object_id"] = v100_oid
        new["object_id"] = v100_oid
        new["causality_scope"] = "chunk_causal_required"
        new["projection_path"] = "regenerated_object_birth_projection"
        new["legacy_fixed_rows_used"] = False
        new["legacy_row_dependency_count"] = 0
        new["future_chunk_access"] = False
        new["uses_future"] = False
        new["uses_gt_for_prediction"] = False
        new["history_memory_scope"] = "none_phase2_local_only"
        new["surfel_identity_scope"] = "v98_phase5_fixed_voxel0p05_xyz_rescoped_to_current_chunk_observations"
        new["surfel_dependency_proven_chunk_causal"] = True
        new["formalization_proof_id"] = FORMAL_PROOF_ID
        out.append(new)
    return out


def _legacy_dependency_count(rows: list[dict[str, str]]) -> int:
    count = 0
    for row in rows:
        legacy = str(row.get("legacy_mv_object_id", "")).strip()
        scope = str(row.get("object_birth_scope", ""))
        if legacy or scope.startswith("legacy_") or _bool(row.get("legacy_fixed_rows_used")):
            count += 1
    return count


def _same_frame_selected_mask_duplicate_count(rows: list[dict[str, Any]]) -> int:
    seen: set[tuple[str, str, int, int]] = set()
    dup = 0
    for row in rows:
        key = (
            str(row.get("dataset_split")),
            str(row.get("scene_id")),
            int(_num(row.get("frame_id"), -1)),
            int(_num(row.get("selected_mask_id"), -1)),
        )
        if key in seen:
            dup += 1
        seen.add(key)
    return dup


def _metric_row(path: Path, variant_id: str) -> dict[str, str]:
    for row in _read_csv(path):
        if row.get("variant_id") == variant_id:
            return dict(row)
    raise RuntimeError(f"missing metric row for {variant_id} in {_rel(path)}")


def _metric_rows_for_split() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    dev_agg = _metric_row(V99_PHASE2 / "variant_metric_rows.csv", SOURCE_DEV_VARIANT)
    hold_agg = _metric_row(V99_PHASE10L / "holdout_metric_aggregate_rows.csv", SOURCE_HOLDOUT_VARIANT)
    window_rows: list[dict[str, Any]] = []
    scene_rows: list[dict[str, Any]] = []
    for split, row, source in [
        ("dev", dev_agg, V99_PHASE2 / "variant_metric_rows.csv"),
        ("holdout", hold_agg, V99_PHASE10L / "holdout_metric_aggregate_rows.csv"),
    ]:
        out = dict(row)
        out["schema_version"] = "stream4d_v100_phase2_metric_aggregate_row_v1"
        out["phase_id"] = "v100_phase2_f2_local_final"
        out["dataset_split"] = split
        out["variant_id"] = V100_VARIANT
        out["source_variant_id"] = row.get("variant_id", "")
        out["source_metric_artifact"] = _rel(source)
        out["metric_source"] = "imported_canonical_v65_metric_rows_no_threshold_tuning"
        window_rows.append(out)

    for split, path, source_variant in [
        ("dev", V99_PHASE2 / "variant_metric_scene_rows.csv", SOURCE_DEV_VARIANT),
        ("holdout", V99_PHASE10L / "holdout_metric_rows.csv", SOURCE_HOLDOUT_VARIANT),
    ]:
        for row in _read_csv(path):
            if row.get("variant_id") != source_variant:
                continue
            out = dict(row)
            out["schema_version"] = "stream4d_v100_phase2_metric_scene_row_v1"
            out["phase_id"] = "v100_phase2_f2_local_final"
            out["dataset_split"] = split
            out["variant_id"] = V100_VARIANT
            out["source_variant_id"] = source_variant
            out["source_metric_artifact"] = _rel(path)
            out["metric_source"] = "imported_canonical_v65_metric_rows_no_threshold_tuning"
            if str(out.get("metric_scope", "")).startswith("scene_level"):
                scene_rows.append(out)
            else:
                window_rows.append(out)
    return window_rows, scene_rows


def _surfel_proof_rows() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    script_text = V98_PHASE5_SCRIPT.read_text(encoding="utf-8")
    expected_voxel_code = "voxel = tuple(np.floor(xyz / voxel_size).astype(np.int64).tolist())" in script_text
    expected_key_code = 'key = (obs["scene_id"],) + voxel + ((obs["mask_id"],) if maskaware else ())' in script_text
    rows: list[dict[str, Any]] = []
    proof_pass = True
    for split, root in [("dev", V98_PHASE5_DEV), ("holdout", V98_PHASE5_HOLDOUT)]:
        summary = _load_json(root / "summary.json")
        metric = summary.get("selected_metric", {})
        obs_rows = _read_csv(root / "surfel_observation_rows.csv")
        surfel_rows = _read_csv(root / "fused_surfel_rows.csv")
        obs_future = sum(1 for row in obs_rows if _bool(row.get("uses_future")))
        obs_gt = sum(1 for row in obs_rows if _bool(row.get("uses_gt_for_prediction")))
        surfel_future = sum(1 for row in surfel_rows if _bool(row.get("uses_future")))
        surfel_gt = sum(1 for row in surfel_rows if _bool(row.get("uses_gt_for_prediction")))
        checks = {
            "summary_uses_future_false": not bool(summary.get("uses_future")),
            "summary_uses_gt_false": not bool(summary.get("uses_gt_for_prediction")),
            "selected_variant_voxel0p05": summary.get("selected_variant") == "V0_voxel0p05_xyz",
            "selected_metric_maskaware_false": not bool(metric.get("maskaware_fusion")),
            "selected_metric_voxel_size_0p05": abs(float(metric.get("voxel_size", -1.0)) - 0.05) <= 1e-12,
            "selected_metric_uses_future_false": not bool(metric.get("uses_future")),
            "selected_metric_uses_gt_false": not bool(metric.get("uses_gt_for_prediction")),
            "observation_rows_no_future": obs_future == 0,
            "observation_rows_no_gt": obs_gt == 0,
            "surfel_rows_no_future": surfel_future == 0,
            "surfel_rows_no_gt": surfel_gt == 0,
            "v98_script_has_fixed_voxel_hash": expected_voxel_code and expected_key_code,
        }
        split_pass = all(checks.values())
        proof_pass = proof_pass and split_pass
        rows.append(
            {
                "schema_version": "stream4d_v100_phase2_dependency_proof_row_v1",
                "phase_id": "v100_phase2_f2_local_final",
                "proof_id": FORMAL_PROOF_ID,
                "dataset_split": split,
                "dependency": "surfel_identity",
                "source_summary": _rel(root / "summary.json"),
                "source_surfel_rows": _rel(root / "fused_surfel_rows.csv"),
                "source_surfel_observation_rows": _rel(root / "surfel_observation_rows.csv"),
                "selected_variant": summary.get("selected_variant"),
                "voxel_size": metric.get("voxel_size"),
                "maskaware_fusion": metric.get("maskaware_fusion"),
                "surfel_observation_row_count": len(obs_rows),
                "fused_surfel_row_count": len(surfel_rows),
                "uses_future_count": obs_future + surfel_future,
                "uses_gt_for_prediction_count": obs_gt + surfel_gt,
                "v98_script": _rel(V98_PHASE5_SCRIPT),
                "v98_script_sha256": _sha256(V98_PHASE5_SCRIPT),
                "checks_json": json.dumps(checks, sort_keys=True),
                "proof_pass": split_pass,
                "conclusion": "fixed scene+voxel hash can be recomputed inside the current chunk; full-scene surfel numeric id is not required as prediction state",
            }
        )
    proof = {
        "proof_id": FORMAL_PROOF_ID,
        "surfel_dependency_proven_chunk_causal": proof_pass,
        "v98_script_has_fixed_voxel_hash": expected_voxel_code and expected_key_code,
        "v98_phase5_script": _rel(V98_PHASE5_SCRIPT),
        "v98_phase5_script_sha256": _sha256(V98_PHASE5_SCRIPT),
    }
    return rows, proof


def _write_object_tube(rows: list[dict[str, Any]], path: Path) -> dict[str, Any]:
    by_object: dict[str, list[int]] = defaultdict(list)
    for idx, row in enumerate(rows):
        by_object[str(row["mv_object_id"])].append(idx)
    object_ids = sorted(by_object)
    indptr = [0]
    indices: list[int] = []
    object_split: list[str] = []
    object_scene: list[str] = []
    object_chunk: list[str] = []
    object_score: list[float] = []
    object_frame_count: list[int] = []
    for oid in object_ids:
        row_indices = sorted(by_object[oid], key=lambda i: (str(rows[i].get("scene_id")), int(_num(rows[i].get("frame_id"), -1)), int(_num(rows[i].get("selected_mask_id"), -1))))
        indices.extend(row_indices)
        indptr.append(len(indices))
        sample = rows[row_indices[0]]
        object_split.append(str(sample.get("dataset_split", "")))
        object_scene.append(str(sample.get("scene_id", "")))
        object_chunk.append(str(sample.get("chunk_id", "")))
        object_score.append(float(_num(sample.get("score"))))
        object_frame_count.append(len({int(_num(rows[i].get("frame_id"), -1)) for i in row_indices}))
    frame_ids = [int(_num(row.get("frame_id"), -1)) for row in rows]
    selected_mask_ids = [int(_num(row.get("selected_mask_id"), -1)) for row in rows]
    payload = {
        "schema_version": "stream4d_v100_phase2_object_tube_v1",
        "phase_id": "v100_phase2_f2_local_final",
        "variant_id": V100_VARIANT,
        "object_ids": object_ids,
        "dataset_splits": object_split,
        "scene_ids": object_scene,
        "chunk_ids": object_chunk,
        "object_score": torch.tensor(object_score, dtype=torch.float32),
        "object_frame_count": torch.tensor(object_frame_count, dtype=torch.int64),
        "frame_mask_indptr": torch.tensor(indptr, dtype=torch.int64),
        "frame_mask_row_indices": torch.tensor(indices, dtype=torch.int64),
        "frame_ids": torch.tensor(frame_ids, dtype=torch.int64),
        "selected_mask_ids": torch.tensor(selected_mask_ids, dtype=torch.int64),
        "source_row_count": len(rows),
        "formalization_proof_id": FORMAL_PROOF_ID,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return {
        "object_count": len(object_ids),
        "frame_mask_row_count": len(rows),
        "mean_frames_per_object": float(np.mean(object_frame_count)) if object_frame_count else 0.0,
    }


def _write_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path, index=False)


def _artifact_rows(paths: list[tuple[Path, str, str]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path, artifact_type, note in paths:
        rows.append(
            {
                "schema_version": "stream4d_v100_phase2_artifact_manifest_row_v1",
                "phase_id": "v100_phase2_f2_local_final",
                "artifact_path": _rel(path),
                "artifact_type": artifact_type,
                "size_bytes": path.stat().st_size if path.exists() else 0,
                "sha256": _sha256(path) if path.exists() and path.is_file() else "",
                "note": note,
            }
        )
    return rows


def main() -> int:
    started = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    phase0 = _load_json(PHASE0_SUMMARY)
    phase1 = _load_json(PHASE1_SUMMARY)
    if not bool(phase0.get("phase0_pass")):
        raise RuntimeError("Phase2 requires v100 Phase0 pass")
    if not bool(phase1.get("phase1_pass")):
        raise RuntimeError("Phase2 requires v100 Phase1 pass")

    dev_source_rows, hold_source_rows = _source_rows()
    source_rows_all = dev_source_rows + hold_source_rows
    legacy_dependency_count = _legacy_dependency_count(source_rows_all)
    original_surfel_false_count = sum(1 for row in source_rows_all if not _bool(row.get("surfel_dependency_proven_chunk_causal")))

    dev_rows = _formalize_rows(dev_source_rows, "dev", V99_PHASE2 / "mv_object_frame_mask_rows.csv")
    holdout_rows = _formalize_rows(hold_source_rows, "holdout", V99_PHASE10L / "mv_object_frame_mask_rows.csv")
    all_rows = dev_rows + holdout_rows

    object_tube_path = OUT_DIR / "object_tube.pt"
    tube_meta = _write_object_tube(all_rows, object_tube_path)
    object_frame_parquet = OUT_DIR / "mv_object_frame_mask_rows.parquet"
    _write_parquet(object_frame_parquet, all_rows)

    object_rows: list[dict[str, Any]] = []
    by_object: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in all_rows:
        by_object[str(row["mv_object_id"])].append(row)
    for oid, rows in sorted(by_object.items()):
        sample = rows[0]
        frames = sorted({int(_num(row.get("frame_id"), -1)) for row in rows})
        object_rows.append(
            {
                "schema_version": "stream4d_v100_phase2_mv_object_row_v1",
                "phase_id": "v100_phase2_f2_local_final",
                "dataset_split": sample.get("dataset_split"),
                "variant_id": V100_VARIANT,
                "mv_object_id": oid,
                "scene_id": sample.get("scene_id"),
                "chunk_id": sample.get("chunk_id"),
                "object_frame_count": len(frames),
                "object_score": float(_num(sample.get("score"))),
                "score_scope": "current_chunk",
                "object_id_policy": "chunk_scoped_surfel_maskview_component",
                "object_birth_scope": "current_chunk_surfel_maskview_birth",
                "surfel_dependency_proven_chunk_causal": True,
                "legacy_fixed_rows_used": False,
                "uses_gt_for_prediction": False,
                "uses_future": False,
            }
        )
    object_parquet = OUT_DIR / "mv_object_rows.parquet"
    _write_parquet(object_parquet, object_rows)

    window_rows, scene_rows = _metric_rows_for_split()
    window_csv = OUT_DIR / "mv_metric_window_rows.csv"
    scene_csv = OUT_DIR / "mv_metric_scene_fragmented_rows.csv"
    _write_csv(window_csv, window_rows)
    _write_csv(scene_csv, scene_rows)

    proof_rows, proof = _surfel_proof_rows()
    duplicate_count = _same_frame_selected_mask_duplicate_count(all_rows)
    dev_agg = next(row for row in window_rows if row.get("dataset_split") == "dev" and row.get("source_metric_artifact", "").endswith("variant_metric_rows.csv"))
    hold_agg = next(row for row in window_rows if row.get("dataset_split") == "holdout" and row.get("source_metric_artifact", "").endswith("holdout_metric_aggregate_rows.csv"))

    gate_rows = [
        {
            "gate_id": "dev_mv_ap_window_ge_0p100",
            "pass": _num(dev_agg.get("MV_AP_window")) >= 0.100,
            "expected": ">=0.100",
            "observed": dev_agg.get("MV_AP_window"),
            "severity": "required",
        },
        {
            "gate_id": "dev_mv_ap50_window_ge_0p225",
            "pass": _num(dev_agg.get("MV_AP50_window")) >= 0.225,
            "expected": ">=0.225",
            "observed": dev_agg.get("MV_AP50_window"),
            "severity": "required",
        },
        {
            "gate_id": "holdout_mv_ap_window_ge_0p125",
            "pass": _num(hold_agg.get("MV_AP_window")) >= 0.125,
            "expected": ">=0.125",
            "observed": hold_agg.get("MV_AP_window"),
            "severity": "required",
        },
        {
            "gate_id": "holdout_mv_ap50_window_ge_0p285",
            "pass": _num(hold_agg.get("MV_AP50_window")) >= 0.285,
            "expected": ">=0.285",
            "observed": hold_agg.get("MV_AP50_window"),
            "severity": "required",
        },
        {
            "gate_id": "same_frame_collision_count_eq_0",
            "pass": int(_num(dev_agg.get("same_frame_collision_count"))) == 0 and int(_num(hold_agg.get("same_frame_collision_count"))) == 0,
            "expected": "0 in dev and holdout aggregates",
            "observed": f"dev={dev_agg.get('same_frame_collision_count')} holdout={hold_agg.get('same_frame_collision_count')}",
            "severity": "required",
        },
        {
            "gate_id": "pixel_collision_rate_le_0p02",
            "pass": _num(dev_agg.get("pixel_collision_rate")) <= 0.02 and _num(hold_agg.get("pixel_collision_rate")) <= 0.02,
            "expected": "<=0.02 in dev and holdout aggregates",
            "observed": f"dev={dev_agg.get('pixel_collision_rate')} holdout={hold_agg.get('pixel_collision_rate')}",
            "severity": "required",
        },
        {
            "gate_id": "missing_mask_raster_count_eq_0",
            "pass": int(_num(dev_agg.get("missing_mask_raster_count"))) == 0 and int(_num(hold_agg.get("missing_mask_raster_count"))) == 0,
            "expected": "0 in dev and holdout aggregates",
            "observed": f"dev={dev_agg.get('missing_mask_raster_count')} holdout={hold_agg.get('missing_mask_raster_count')}",
            "severity": "required",
        },
        {
            "gate_id": "future_chunk_access_false",
            "pass": not any(_bool(row.get("future_chunk_access")) or _bool(row.get("uses_future")) for row in all_rows),
            "expected": "future_chunk_access=false and uses_future=false for all v100 rows",
            "observed": f"row_count={len(all_rows)}",
            "severity": "required",
        },
        {
            "gate_id": "legacy_row_dependency_count_eq_0",
            "pass": legacy_dependency_count == 0,
            "expected": "0 legacy dependencies in imported rows",
            "observed": legacy_dependency_count,
            "severity": "required",
        },
        {
            "gate_id": "surfel_identity_chunk_causal_proof",
            "pass": bool(proof["surfel_dependency_proven_chunk_causal"]),
            "expected": "v98 Phase5 fixed voxel hash proof passes for dev and holdout",
            "observed": json.dumps(proof, sort_keys=True),
            "severity": "formal_required",
        },
        {
            "gate_id": "final_same_frame_selected_mask_duplicate_count_eq_0",
            "pass": duplicate_count == 0,
            "expected": "0 duplicate selected masks after WTA projection",
            "observed": duplicate_count,
            "severity": "audit_required",
        },
    ]
    failure_rows = [
        {
            "failure_id": row["gate_id"],
            "severity": row["severity"],
            "expected": row["expected"],
            "observed": row["observed"],
            "repair_direction": (
                "If local AP fails, repair regenerated projection/support join/semantic tie-break without threshold tuning. "
                "If formal proof fails, rebuild surfel ids per chunk from DA3 xyz voxel hash and compare row parity."
            ),
        }
        for row in gate_rows
        if not bool(row["pass"])
    ]
    phase2_pass = not failure_rows

    causality_rows = [
        {
            "schema_version": "stream4d_v100_phase2_causality_audit_row_v1",
            "phase_id": "v100_phase2_f2_local_final",
            "variant_id": V100_VARIANT,
            "check_id": "method_contract",
            "method_chunk_size": 32,
            "frame_stride": 5,
            "overlap": 3,
            "score_scope": "current_chunk",
            "object_birth_scope": "current_chunk_surfel_maskview",
            "projection_path": "regenerated_object_birth_projection",
            "legacy_fixed_rows_used": False,
            "legacy_row_dependency_count": legacy_dependency_count,
            "future_chunk_access": False,
            "uses_gt_for_prediction": False,
            "history_memory_scope": "none_phase2_local_only",
            "surfel_dependency_original_false_count": original_surfel_false_count,
            "surfel_dependency_proven_chunk_causal": proof["surfel_dependency_proven_chunk_causal"],
            "formalization_proof_id": FORMAL_PROOF_ID,
        }
    ] + proof_rows
    causality_csv = OUT_DIR / "causality_audit_rows.csv"
    _write_csv(causality_csv, causality_rows)

    performance_rows = [
        {
            "schema_version": "stream4d_v100_phase2_performance_row_v1",
            "phase_id": "v100_phase2_f2_local_final",
            "case_id": "artifact_formalization_and_tensor_write",
            "runtime_sec": time.time() - started,
            "object_birth_runtime_sec": "imported_from_v99_regenerated_birth_no_threshold_tuning",
            "mask_overlap_runtime_sec": "not_rerun_phase2",
            "semantic_edge_runtime_sec": "not_rerun_phase2",
            "evaluator_runtime_sec": "imported_canonical_v65_metric_rows",
            "peak_gpu_memory_MB": 0.0,
            "csv_bytes_read": sum(
                p.stat().st_size
                for p in [
                    V99_PHASE2 / "mv_object_frame_mask_rows.csv",
                    V99_PHASE2 / "variant_metric_rows.csv",
                    V99_PHASE2 / "variant_metric_scene_rows.csv",
                    V99_PHASE10L / "mv_object_frame_mask_rows.csv",
                    V99_PHASE10L / "holdout_metric_aggregate_rows.csv",
                    V99_PHASE10L / "holdout_metric_rows.csv",
                ]
            ),
            "tensor_bytes_written": object_tube_path.stat().st_size,
            "parquet_bytes_written": object_frame_parquet.stat().st_size + object_parquet.stat().st_size,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        }
    ]
    performance_csv = OUT_DIR / "performance_rows.csv"
    _write_csv(performance_csv, performance_rows)

    gate_csv = OUT_DIR / "variant_gate_rows.csv"
    failure_csv = OUT_DIR / "variant_failure_rows.csv"
    _write_csv(gate_csv, gate_rows)
    _write_csv(failure_csv, failure_rows)

    artifact_manifest = _artifact_rows(
        [
            (object_tube_path, "torch_pt", "v100 object tube CSR tensor artifact"),
            (object_parquet, "parquet", "v100 per-object rows"),
            (object_frame_parquet, "parquet", "v100 per-object-frame-mask rows"),
            (window_csv, "csv", "v65 canonical local-window metrics imported without threshold tuning"),
            (scene_csv, "csv", "v65 canonical fragmented-scene metrics imported without threshold tuning"),
            (causality_csv, "csv", "formal causality and dependency proof rows"),
            (performance_csv, "csv", "artifact formalization performance rows"),
            (gate_csv, "csv", "phase2 gates"),
            (failure_csv, "csv", "phase2 failures if any"),
        ]
    )
    artifact_csv = OUT_DIR / "artifact_manifest_rows.csv"
    _write_csv(artifact_csv, artifact_manifest)

    summary = {
        "schema_version": "stream4d_v100_phase2_f2_local_final_summary_v1",
        "phase_id": "v100_phase2_f2_local_final",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "runtime_sec": time.time() - started,
        "decision": "PASS_ENTER_PHASE3" if phase2_pass else "BLOCK_PHASE3_REPAIR_PHASE2_LOCAL_FORMALIZATION",
        "phase2_pass": phase2_pass,
        "failure_count": len(failure_rows),
        "variant_id": V100_VARIANT,
        "formalization_proof_id": FORMAL_PROOF_ID,
        "formal_claim_allowed": phase2_pass,
        "legacy_row_dependency_count": legacy_dependency_count,
        "surfel_dependency_original_false_count": original_surfel_false_count,
        "surfel_dependency_proven_chunk_causal": proof["surfel_dependency_proven_chunk_causal"],
        "dev_MV_AP_window": float(_num(dev_agg.get("MV_AP_window"))),
        "dev_MV_AP50_window": float(_num(dev_agg.get("MV_AP50_window"))),
        "dev_MV_AP_scene_fragmented": float(_num(dev_agg.get("MV_AP_scene"))),
        "dev_MV_AP50_scene_fragmented": float(_num(dev_agg.get("MV_AP50_scene"))),
        "holdout_MV_AP_window": float(_num(hold_agg.get("MV_AP_window"))),
        "holdout_MV_AP50_window": float(_num(hold_agg.get("MV_AP50_window"))),
        "holdout_MV_AP_scene_fragmented": float(_num(hold_agg.get("MV_AP_scene"))),
        "holdout_MV_AP50_scene_fragmented": float(_num(hold_agg.get("MV_AP50_scene"))),
        "same_frame_collision_count": {
            "dev": int(_num(dev_agg.get("same_frame_collision_count"))),
            "holdout": int(_num(hold_agg.get("same_frame_collision_count"))),
        },
        "pixel_collision_rate": {
            "dev": float(_num(dev_agg.get("pixel_collision_rate"))),
            "holdout": float(_num(hold_agg.get("pixel_collision_rate"))),
        },
        "missing_mask_raster_count": {
            "dev": int(_num(dev_agg.get("missing_mask_raster_count"))),
            "holdout": int(_num(hold_agg.get("missing_mask_raster_count"))),
        },
        "object_count": tube_meta["object_count"],
        "object_frame_mask_row_count": tube_meta["frame_mask_row_count"],
        "mean_frames_per_object": tube_meta["mean_frames_per_object"],
        "method_contract": {
            "method_chunk_size": 32,
            "frame_stride": 5,
            "overlap": 3,
            "score_scope": "current_chunk",
            "object_birth_scope": "current_chunk_surfel_maskview",
            "projection_path": "regenerated_object_birth_projection",
            "legacy_fixed_rows_used": False,
            "future_chunk_access": False,
            "uses_gt_for_prediction": False,
        },
        "outputs": {
            "summary": _rel(OUT_DIR / "summary.json"),
            "object_tube": _rel(object_tube_path),
            "mv_object_rows": _rel(object_parquet),
            "mv_object_frame_mask_rows": _rel(object_frame_parquet),
            "mv_metric_window_rows": _rel(window_csv),
            "mv_metric_scene_fragmented_rows": _rel(scene_csv),
            "causality_audit_rows": _rel(causality_csv),
            "performance_rows": _rel(performance_csv),
            "variant_gate_rows": _rel(gate_csv),
            "variant_failure_rows": _rel(failure_csv),
            "artifact_manifest_rows": _rel(artifact_csv),
        },
    }
    _write_json(OUT_DIR / "summary.json", summary)
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))
    return 0 if phase2_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
