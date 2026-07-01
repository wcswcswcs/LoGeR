#!/usr/bin/env python3
"""Build a v100 Phase2 overlap3 local repair artifact.

v100 Phase2 imported strong v99 F2 rows, but the materialized rows used
non-overlapping chunks while the plan contract requires chunk32/overlap3.
This script keeps the old Phase2 artifact intact and rebuilds the local F2
surfel-maskview birth with explicit frame_to_chunks overlap membership,
following the v99 Phase10O repair pattern.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
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

from tools import build_v99_phase1_f2_base_reproduction as p1  # noqa: E402
from tools import build_v99_phase10k_holdout_chunk_object_birth_sweep as p10k  # noqa: E402
from tools import build_v99_phase10o_overlap3_scene_stitch_repair as p10o  # noqa: E402


AUDIT_ROOT = STREAM3D_ROOT / "outputs/audit"
OUT_DIR = AUDIT_ROOT / "v100_phase2c_overlap3_local_repair"
PHASE0_SUMMARY = AUDIT_ROOT / "v100_phase0_contract/summary.json"
PHASE2_SUMMARY = AUDIT_ROOT / "v100_phase2_f2_local_final/summary.json"

CHUNK_SIZE = 32
OVERLAP = 3
STEP = CHUNK_SIZE - OVERLAP
V100_VARIANT = "F2_v100_chunk32_overlap3_surfel_maskview_thr018_p2d2"
FORMAL_PROOF_ID = "v100_phase2c_overlap3_fixed_voxel_hash_rescope_reuses_phase2_proof_v1"
EPS = 1e-4

DEV_INPUTS = {
    "SOURCE_ROWS": p1.SOURCE_ROWS,
    "RADIO_MASK_FEATURES": p1.RADIO_MASK_FEATURES,
    "SURFEL_ROWS": p1.SURFEL_ROWS,
    "SURFEL_OBS_ROWS": p1.SURFEL_OBS_ROWS,
    "SURFEL_SUMMARY": p1.SURFEL_SUMMARY,
}
HOLDOUT_INPUTS = {
    "SOURCE_ROWS": p10k.HOLDOUT_SOURCE_ROWS,
    "RADIO_MASK_FEATURES": p10k.HOLDOUT_RADIO_MASK_FEATURES,
    "SURFEL_ROWS": p10k.HOLDOUT_SURFEL_ROWS,
    "SURFEL_OBS_ROWS": p10k.HOLDOUT_SURFEL_OBS_ROWS,
    "SURFEL_SUMMARY": p10k.HOLDOUT_SURFEL_SUMMARY,
}


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


def _write_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path, index=False)


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


def _set_inputs(inputs: dict[str, Path]) -> None:
    p1.SOURCE_ROWS = inputs["SOURCE_ROWS"]
    p1.RADIO_MASK_FEATURES = inputs["RADIO_MASK_FEATURES"]
    p1.SURFEL_ROWS = inputs["SURFEL_ROWS"]
    p1.SURFEL_OBS_ROWS = inputs["SURFEL_OBS_ROWS"]
    p1.SURFEL_SUMMARY = inputs["SURFEL_SUMMARY"]


def _norm(values: dict[str, float]) -> dict[str, float]:
    vals = [float(v) for v in values.values() if math.isfinite(float(v))]
    if not vals:
        return {key: 0.0 for key in values}
    lo = min(vals)
    hi = max(vals)
    if hi - lo <= 1e-12:
        return {key: 0.0 for key in values}
    return {key: (float(val) - lo) / (hi - lo) for key, val in values.items()}


def _semantic_norm_by_object(rows: list[dict[str, Any]]) -> dict[str, float]:
    features, _tau = p1._load_radio_residual_features()
    by_object: dict[str, list[np.ndarray]] = defaultdict(list)
    for row in rows:
        key = (str(row["scene_id"]), int(row["frame_id"]), int(row["selected_mask_id"]))
        feat = features.get(key)
        if feat is not None:
            by_object[str(row["mv_object_id"])].append(feat)
    raw: dict[str, float] = {}
    for oid, vals in by_object.items():
        if len(vals) < 2:
            raw[oid] = 0.0
            continue
        stack = np.stack(vals).astype(np.float32)
        centroid = p1._normalize_rows(np.mean(stack, axis=0, keepdims=True))[0]
        raw[oid] = float(np.mean([p1._cosine(row, centroid) for row in stack]))
    for row in rows:
        raw.setdefault(str(row["mv_object_id"]), 0.0)
    return _norm(raw)


def _apply_p2d2_score(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    frames_by_object: dict[str, set[tuple[str, int]]] = defaultdict(set)
    for row in rows:
        frames_by_object[str(row["mv_object_id"])].add((str(row["scene_id"]), int(row["frame_id"])))
    semantic_norm = _semantic_norm_by_object(rows)
    out: list[dict[str, Any]] = []
    for row in rows:
        oid = str(row["mv_object_id"])
        frame_count_score = len(frames_by_object[oid]) / float(CHUNK_SIZE)
        new = dict(row)
        new["variant_id"] = V100_VARIANT
        new["variant"] = V100_VARIANT
        new["score"] = float(frame_count_score + EPS * semantic_norm.get(oid, 0.0))
        new["score_scope"] = "current_chunk"
        new["score_policy"] = "current_chunk_frame_count_over_32_plus_1e-4_semantic_consistency_tiebreak"
        new["v100_parent_variant_id"] = row.get("variant_id", "")
        new["v100_frame_count_score"] = frame_count_score
        new["v100_semantic_norm"] = semantic_norm.get(oid, 0.0)
        new["uses_gt_for_prediction"] = False
        new["uses_future"] = False
        out.append(new)
    return out


def _canonicalize_rows(rows: list[dict[str, Any]], split: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    identity_scope = (
        "v98_phase5_fixed_voxel0p05_xyz_rescoped_to_current_overlap_chunk_observations"
        if split == "dev"
        else "v98_phase13_holdout_phase5_fixed_voxel0p05_xyz_rescoped_to_current_overlap_chunk_observations"
    )
    for row in rows:
        source_oid = str(row["mv_object_id"])
        oid = f"{split}:{source_oid}"
        new = dict(row)
        new["schema_version"] = "stream4d_v100_phase2c_mv_object_frame_mask_row_v1"
        new["phase_id"] = "v100_phase2c_overlap3_local_repair"
        new["dataset_split"] = split
        new["variant_id"] = V100_VARIANT
        new["variant"] = V100_VARIANT
        new["source_mv_object_id"] = source_oid
        new["mv_object_id"] = oid
        new["object_id"] = oid
        new["method_chunk_size"] = CHUNK_SIZE
        new["method_chunk_overlap"] = OVERLAP
        new["frame_stride"] = p1.FRAME_STRIDE
        new["causality_scope"] = "chunk_causal_overlap3_required"
        new["projection_path"] = "regenerated_overlap3_object_birth_primary_emit"
        new["legacy_fixed_rows_used"] = False
        new["legacy_row_dependency_count"] = 0
        new["future_chunk_access"] = False
        new["uses_future"] = False
        new["uses_gt_for_prediction"] = False
        new["history_memory_scope"] = "none_phase2c_local_only"
        new["object_birth_scope"] = "current_chunk32_overlap3_surfel_maskview_birth"
        new["object_id_policy"] = "chunk32_overlap3_primary_emit_chunk_scoped_identity"
        new["surfel_identity_scope"] = identity_scope
        new["surfel_dependency_proven_chunk_causal"] = True
        new["formalization_proof_id"] = FORMAL_PROOF_ID
        out.append(new)
    return out


def _object_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_object: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_object[str(row["mv_object_id"])].append(row)
    out: list[dict[str, Any]] = []
    for oid, vals in sorted(by_object.items()):
        sample = vals[0]
        frames = sorted({int(row["frame_id"]) for row in vals})
        out.append(
            {
                "schema_version": "stream4d_v100_phase2c_mv_object_row_v1",
                "phase_id": "v100_phase2c_overlap3_local_repair",
                "dataset_split": sample.get("dataset_split"),
                "variant_id": V100_VARIANT,
                "mv_object_id": oid,
                "source_mv_object_id": sample.get("source_mv_object_id", ""),
                "scene_id": sample.get("scene_id"),
                "chunk_id": sample.get("chunk_id"),
                "object_frame_count": len(frames),
                "object_score": float(_num(sample.get("score"))),
                "score_scope": "current_chunk",
                "object_id_policy": "chunk32_overlap3_primary_emit_chunk_scoped_identity",
                "object_birth_scope": "current_chunk32_overlap3_surfel_maskview_birth",
                "surfel_dependency_proven_chunk_causal": True,
                "formalization_proof_id": FORMAL_PROOF_ID,
                "legacy_fixed_rows_used": False,
                "uses_gt_for_prediction": False,
                "uses_future": False,
            }
        )
    return out


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
        row_indices = sorted(
            by_object[oid],
            key=lambda i: (
                str(rows[i].get("scene_id")),
                str(rows[i].get("chunk_id")),
                int(rows[i].get("frame_id", -1)),
                int(rows[i].get("selected_mask_id", -1)),
            ),
        )
        indices.extend(row_indices)
        indptr.append(len(indices))
        sample = rows[row_indices[0]]
        object_split.append(str(sample.get("dataset_split", "")))
        object_scene.append(str(sample.get("scene_id", "")))
        object_chunk.append(str(sample.get("chunk_id", "")))
        object_score.append(float(_num(sample.get("score"))))
        object_frame_count.append(len({int(rows[i].get("frame_id", -1)) for i in row_indices}))
    payload = {
        "schema_version": "stream4d_v100_phase2c_object_tube_v1",
        "phase_id": "v100_phase2c_overlap3_local_repair",
        "variant_id": V100_VARIANT,
        "object_ids": object_ids,
        "dataset_splits": object_split,
        "scene_ids": object_scene,
        "chunk_ids": object_chunk,
        "object_score": torch.tensor(object_score, dtype=torch.float32),
        "object_frame_count": torch.tensor(object_frame_count, dtype=torch.int64),
        "frame_mask_indptr": torch.tensor(indptr, dtype=torch.int64),
        "frame_mask_row_indices": torch.tensor(indices, dtype=torch.int64),
        "frame_ids": torch.tensor([int(row.get("frame_id", -1)) for row in rows], dtype=torch.int64),
        "selected_mask_ids": torch.tensor([int(row.get("selected_mask_id", -1)) for row in rows], dtype=torch.int64),
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


def _split_artifact_inputs(split: str) -> dict[str, str]:
    inputs = DEV_INPUTS if split == "dev" else HOLDOUT_INPUTS
    return {key.lower(): _rel(value) for key, value in inputs.items()}


def _run_split(split: str) -> dict[str, Any]:
    _set_inputs(DEV_INPUTS if split == "dev" else HOLDOUT_INPUTS)
    scope = p10o._build_overlap3_scope()
    eval_scope = p10o._eval_scope_from_overlap(scope)
    raw_rows, _raw_object_rows, birth_stats = p10o._build_overlap_surfel_rows(scope)
    scored_internal = _apply_p2d2_score(raw_rows)
    primary_rows = p10o._primary_emit_rows(scored_internal, scope)
    rows = _canonicalize_rows(primary_rows, split)
    metrics, frame_rows = p1._evaluate_variant(V100_VARIANT, rows, eval_scope)
    aggregate_rows = p1._aggregate_metrics(metrics)
    if len(aggregate_rows) != 1:
        raise RuntimeError(f"expected one aggregate row for {split}, got {len(aggregate_rows)}")
    agg = dict(aggregate_rows[0])
    agg["schema_version"] = "stream4d_v100_phase2c_metric_aggregate_row_v1"
    agg["phase_id"] = "v100_phase2c_overlap3_local_repair"
    agg["dataset_split"] = split
    agg["variant_id"] = V100_VARIANT
    agg["metric_source"] = "fresh_v65_eval_on_overlap3_primary_emit_rows"
    for row in metrics:
        row["dataset_split"] = split
        row["phase_id"] = "v100_phase2c_overlap3_local_repair"
    for row in frame_rows:
        row["dataset_split"] = split
        row["phase_id"] = "v100_phase2c_overlap3_local_repair"
    overlap_rows: list[dict[str, Any]] = []
    for row in scope["overlap_pair_rows"]:
        new = dict(row)
        new["schema_version"] = "stream4d_v100_phase2c_overlap_pair_row_v1"
        new["phase_id"] = "v100_phase2c_overlap3_local_repair"
        new["dataset_split"] = split
        new["overlap_contract_pass"] = int(new["shared_frame_count"]) == OVERLAP
        overlap_rows.append(new)
    return {
        "split": split,
        "rows": rows,
        "object_rows": _object_rows(rows),
        "metric_rows": metrics,
        "frame_rows": frame_rows,
        "aggregate": agg,
        "overlap_rows": overlap_rows,
        "birth_stats": birth_stats,
        "source_uses_future": bool(scope.get("source_uses_future")),
        "source_uses_gt_for_prediction": bool(scope.get("source_uses_gt_for_prediction")),
        "artifact_inputs": _split_artifact_inputs(split),
    }


def _artifact_rows(paths: list[tuple[Path, str, str]]) -> list[dict[str, Any]]:
    return [
        {
            "schema_version": "stream4d_v100_phase2c_artifact_manifest_row_v1",
            "phase_id": "v100_phase2c_overlap3_local_repair",
            "artifact_path": _rel(path),
            "artifact_type": kind,
            "size_bytes": path.stat().st_size if path.exists() else 0,
            "sha256": _sha256(path) if path.exists() and path.is_file() else "",
            "note": note,
        }
        for path, kind, note in paths
    ]


def main() -> int:
    started = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    phase0 = json.loads(PHASE0_SUMMARY.read_text(encoding="utf-8"))
    phase2 = json.loads(PHASE2_SUMMARY.read_text(encoding="utf-8"))
    if not bool(phase0.get("phase0_pass")):
        raise RuntimeError("Phase2c requires v100 Phase0 pass")
    if not bool(phase2.get("surfel_dependency_proven_chunk_causal")):
        raise RuntimeError("Phase2c requires the v100 Phase2 fixed-voxel rescope proof to pass")

    split_payloads = [_run_split("dev"), _run_split("holdout")]
    all_rows = [row for payload in split_payloads for row in payload["rows"]]
    all_object_rows = [row for payload in split_payloads for row in payload["object_rows"]]
    metric_rows = [row for payload in split_payloads for row in payload["metric_rows"]]
    frame_rows = [row for payload in split_payloads for row in payload["frame_rows"]]
    aggregate_rows = [payload["aggregate"] for payload in split_payloads]
    overlap_rows = [row for payload in split_payloads for row in payload["overlap_rows"]]
    agg_by_split = {row["dataset_split"]: row for row in aggregate_rows}

    object_tube_path = OUT_DIR / "object_tube.pt"
    tube_meta = _write_object_tube(all_rows, object_tube_path)
    object_frame_parquet = OUT_DIR / "mv_object_frame_mask_rows.parquet"
    object_parquet = OUT_DIR / "mv_object_rows.parquet"
    _write_parquet(object_frame_parquet, all_rows)
    _write_parquet(object_parquet, all_object_rows)

    window_csv = OUT_DIR / "mv_metric_window_rows.csv"
    scene_csv = OUT_DIR / "mv_metric_scene_fragmented_rows.csv"
    aggregate_csv = OUT_DIR / "variant_metric_rows.csv"
    frame_csv = OUT_DIR / "frame_eval_rows.csv"
    overlap_csv = OUT_DIR / "overlap_pair_rows.csv"
    _write_csv(window_csv, metric_rows)
    _write_csv(scene_csv, metric_rows)
    _write_csv(aggregate_csv, aggregate_rows)
    _write_csv(frame_csv, frame_rows)
    _write_csv(overlap_csv, overlap_rows)

    dev = agg_by_split["dev"]
    hold = agg_by_split["holdout"]
    overlap_pass = bool(overlap_rows) and all(int(row["shared_frame_count"]) == OVERLAP for row in overlap_rows)
    safety_pass = (
        int(_num(dev.get("same_frame_collision_count"), 1)) == 0
        and int(_num(hold.get("same_frame_collision_count"), 1)) == 0
        and int(_num(dev.get("missing_mask_raster_count"), 1)) == 0
        and int(_num(hold.get("missing_mask_raster_count"), 1)) == 0
        and not any(payload["source_uses_future"] or payload["source_uses_gt_for_prediction"] for payload in split_payloads)
        and not any(_bool(row.get("uses_future")) or _bool(row.get("uses_gt_for_prediction")) for row in all_rows)
    )
    dev_gate = _num(dev.get("MV_AP_window")) >= 0.100 and _num(dev.get("MV_AP50_window")) >= 0.225
    hold_gate = _num(hold.get("MV_AP_window")) >= 0.125 and _num(hold.get("MV_AP50_window")) >= 0.285
    proof_pass = bool(phase2.get("surfel_dependency_proven_chunk_causal")) and bool(phase2.get("formal_claim_allowed"))
    phase2c_pass = bool(overlap_pass and safety_pass and dev_gate and hold_gate and proof_pass)

    gate_rows = [
        {
            "gate_id": "overlap3_contract_shared_frames",
            "pass": overlap_pass,
            "expected": "every adjacent internal chunk pair shared_frame_count=3",
            "observed": sorted({int(row["shared_frame_count"]) for row in overlap_rows}),
            "severity": "formal_contract_required",
        },
        {
            "gate_id": "dev_mv_ap_window_ge_0p100",
            "pass": _num(dev.get("MV_AP_window")) >= 0.100,
            "expected": ">=0.100",
            "observed": dev.get("MV_AP_window"),
            "severity": "required",
        },
        {
            "gate_id": "dev_mv_ap50_window_ge_0p225",
            "pass": _num(dev.get("MV_AP50_window")) >= 0.225,
            "expected": ">=0.225",
            "observed": dev.get("MV_AP50_window"),
            "severity": "required",
        },
        {
            "gate_id": "holdout_mv_ap_window_ge_0p125",
            "pass": _num(hold.get("MV_AP_window")) >= 0.125,
            "expected": ">=0.125",
            "observed": hold.get("MV_AP_window"),
            "severity": "required",
        },
        {
            "gate_id": "holdout_mv_ap50_window_ge_0p285",
            "pass": _num(hold.get("MV_AP50_window")) >= 0.285,
            "expected": ">=0.285",
            "observed": hold.get("MV_AP50_window"),
            "severity": "required",
        },
        {
            "gate_id": "safety_no_gt_no_future_no_missing_no_collision",
            "pass": safety_pass,
            "expected": "uses_gt=false uses_future=false missing_mask=0 same_frame_collision=0",
            "observed": f"dev_collision={dev.get('same_frame_collision_count')} hold_collision={hold.get('same_frame_collision_count')} dev_missing={dev.get('missing_mask_raster_count')} hold_missing={hold.get('missing_mask_raster_count')} source_gt_future={[{'split': p['split'], 'gt': p['source_uses_gt_for_prediction'], 'future': p['source_uses_future']} for p in split_payloads]}",
            "severity": "required",
        },
        {
            "gate_id": "surfel_identity_fixed_voxel_rescope_proof_reused",
            "pass": proof_pass,
            "expected": "v100 Phase2 fixed-voxel rescope proof passes and formal_claim_allowed=true",
            "observed": f"phase2_surfel_dependency_proven_chunk_causal={phase2.get('surfel_dependency_proven_chunk_causal')} phase2_formal_claim_allowed={phase2.get('formal_claim_allowed')}",
            "severity": "formal_required",
        },
    ]
    failure_rows = [
        {
            "schema_version": "stream4d_v100_phase2c_failure_row_v1",
            "phase_id": "v100_phase2c_overlap3_local_repair",
            "failure_id": row["gate_id"],
            "severity": row["severity"],
            "expected": row["expected"],
            "observed": row["observed"],
            "repair_direction": (
                "If overlap contract passes but local AP fails, inspect primary emit, duplicate suppression, and "
                "whether overlap context changed object scores; do not tune thresholds without a pre-registered score branch."
            ),
        }
        for row in gate_rows
        if not bool(row["pass"])
    ]
    gate_csv = OUT_DIR / "variant_gate_rows.csv"
    failure_csv = OUT_DIR / "variant_failure_rows.csv"
    _write_csv(gate_csv, gate_rows)
    _write_csv(failure_csv, failure_rows)

    artifact_csv = OUT_DIR / "artifact_manifest_rows.csv"
    artifact_manifest = _artifact_rows(
        [
            (object_tube_path, "torch_pt", "v100 overlap3 object tube CSR tensor artifact"),
            (object_frame_parquet, "parquet", "v100 overlap3 per-object-frame-mask rows"),
            (object_parquet, "parquet", "v100 overlap3 per-object rows"),
            (aggregate_csv, "csv", "v65 aggregate metrics for dev and holdout"),
            (window_csv, "csv", "v65 per-scene/window metric rows"),
            (scene_csv, "csv", "same v65 metric rows, retained for Phase2 schema compatibility"),
            (frame_csv, "csv", "v65 frame eval rows"),
            (overlap_csv, "csv", "overlap3 internal chunk pair evidence"),
            (gate_csv, "csv", "phase2c gates"),
            (failure_csv, "csv", "phase2c failures if any"),
        ]
    )
    _write_csv(artifact_csv, artifact_manifest)

    summary = {
        "schema_version": "stream4d_v100_phase2c_overlap3_local_repair_summary_v1",
        "phase_id": "v100_phase2c_overlap3_local_repair",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "runtime_sec": time.time() - started,
        "decision": "PASS_ENTER_PHASE3_OVERLAP3_LOCAL" if phase2c_pass else "BLOCK_PHASE3_REPAIR_PHASE2C_OVERLAP3_LOCAL",
        "phase2c_pass": phase2c_pass,
        "failure_count": len(failure_rows),
        "variant_id": V100_VARIANT,
        "formalization_proof_id": FORMAL_PROOF_ID,
        "formal_claim_allowed": phase2c_pass,
        "overlap_contract_gate_pass": overlap_pass,
        "safety_gate_pass": safety_pass,
        "dev_gate_pass": dev_gate,
        "holdout_gate_pass": hold_gate,
        "surfel_dependency_proven_chunk_causal": proof_pass,
        "proof_reuse_source": _rel(PHASE2_SUMMARY),
        "dev_MV_AP_window": float(_num(dev.get("MV_AP_window"))),
        "dev_MV_AP50_window": float(_num(dev.get("MV_AP50_window"))),
        "dev_MV_AP_scene_fragmented": float(_num(dev.get("MV_AP_scene"))),
        "dev_MV_AP50_scene_fragmented": float(_num(dev.get("MV_AP50_scene"))),
        "holdout_MV_AP_window": float(_num(hold.get("MV_AP_window"))),
        "holdout_MV_AP50_window": float(_num(hold.get("MV_AP50_window"))),
        "holdout_MV_AP_scene_fragmented": float(_num(hold.get("MV_AP_scene"))),
        "holdout_MV_AP50_scene_fragmented": float(_num(hold.get("MV_AP50_scene"))),
        "same_frame_collision_count": {
            "dev": int(_num(dev.get("same_frame_collision_count"))),
            "holdout": int(_num(hold.get("same_frame_collision_count"))),
        },
        "pixel_collision_rate": {
            "dev": float(_num(dev.get("pixel_collision_rate"))),
            "holdout": float(_num(hold.get("pixel_collision_rate"))),
        },
        "missing_mask_raster_count": {
            "dev": int(_num(dev.get("missing_mask_raster_count"))),
            "holdout": int(_num(hold.get("missing_mask_raster_count"))),
        },
        "object_count": tube_meta["object_count"],
        "object_frame_mask_row_count": tube_meta["frame_mask_row_count"],
        "mean_frames_per_object": tube_meta["mean_frames_per_object"],
        "overlap_transition_count": len(overlap_rows),
        "min_observed_overlap": min([int(row["shared_frame_count"]) for row in overlap_rows] or [0]),
        "max_observed_overlap": max([int(row["shared_frame_count"]) for row in overlap_rows] or [0]),
        "method_contract": {
            "method_chunk_size": CHUNK_SIZE,
            "frame_stride": p1.FRAME_STRIDE,
            "overlap": OVERLAP,
            "score_scope": "current_chunk",
            "object_birth_scope": "current_chunk32_overlap3_surfel_maskview",
            "projection_path": "regenerated_overlap3_object_birth_primary_emit",
            "eval_emit_policy": "earliest_chunk_owns_frame_overlap_context_not_double_counted",
            "legacy_fixed_rows_used": False,
            "future_chunk_access": False,
            "uses_gt_for_prediction": False,
        },
        "birth_stats": {payload["split"]: payload["birth_stats"] for payload in split_payloads},
        "artifact_inputs": {payload["split"]: payload["artifact_inputs"] for payload in split_payloads},
        "outputs": {
            "summary": _rel(OUT_DIR / "summary.json"),
            "object_tube": _rel(object_tube_path),
            "mv_object_rows": _rel(object_parquet),
            "mv_object_frame_mask_rows": _rel(object_frame_parquet),
            "mv_metric_window_rows": _rel(window_csv),
            "mv_metric_scene_fragmented_rows": _rel(scene_csv),
            "variant_metric_rows": _rel(aggregate_csv),
            "frame_eval_rows": _rel(frame_csv),
            "overlap_pair_rows": _rel(overlap_csv),
            "variant_gate_rows": _rel(gate_csv),
            "variant_failure_rows": _rel(failure_csv),
            "artifact_manifest_rows": _rel(artifact_csv),
        },
    }
    _write_json(OUT_DIR / "summary.json", summary)
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))
    return 0 if phase2c_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
