#!/usr/bin/env python3
"""Build v100 Phase3 scene fragmentation audit."""

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
AUDIT_ROOT = STREAM3D_ROOT / "outputs/audit"
PHASE2_DIR = AUDIT_ROOT / "v100_phase2_f2_local_final"
OUT_DIR = AUDIT_ROOT / "v100_phase3_scene_fragmentation_audit"


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


def _load_phase2_summary() -> dict[str, Any]:
    return json.loads((PHASE2_DIR / "summary.json").read_text(encoding="utf-8"))


def _metric_lookup() -> tuple[dict[tuple[str, str], dict[str, str]], dict[str, dict[str, str]]]:
    scene_lookup: dict[tuple[str, str], dict[str, str]] = {}
    for row in _read_csv(PHASE2_DIR / "mv_metric_scene_fragmented_rows.csv"):
        if str(row.get("metric_scope", "")).startswith("scene_level"):
            scene_lookup[(row.get("dataset_split", ""), row.get("scene_id", ""))] = row
    agg_lookup: dict[str, dict[str, str]] = {}
    for row in _read_csv(PHASE2_DIR / "mv_metric_window_rows.csv"):
        if row.get("source_metric_artifact", "").endswith(("variant_metric_rows.csv", "holdout_metric_aggregate_rows.csv")):
            agg_lookup[row.get("dataset_split", "")] = row
    return scene_lookup, agg_lookup


def _object_chunk_rows(df: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for (split, scene, oid), grp in df.groupby(["dataset_split", "scene_id", "mv_object_id"], dropna=False):
        chunks = sorted(str(v) for v in grp["chunk_id"].dropna().unique())
        frames = sorted(int(v) for v in grp["frame_id"].dropna().unique())
        rows.append(
            {
                "schema_version": "stream4d_v100_phase3_object_chunk_row_v1",
                "phase_id": "v100_phase3_scene_fragmentation_audit",
                "dataset_split": split,
                "scene_id": scene,
                "mv_object_id": oid,
                "chunk_count": len(chunks),
                "chunks": ";".join(chunks),
                "frame_count": len(frames),
                "first_frame": min(frames) if frames else "",
                "last_frame": max(frames) if frames else "",
                "crosses_multiple_chunks": len(chunks) > 1,
            }
        )
    return rows


def main() -> int:
    started = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    phase2 = _load_phase2_summary()
    if not bool(phase2.get("phase2_pass")):
        raise RuntimeError("Phase3 requires v100 Phase2 pass")

    frame_mask_path = PHASE2_DIR / "mv_object_frame_mask_rows.parquet"
    object_tube_path = PHASE2_DIR / "object_tube.pt"
    df = pd.read_parquet(frame_mask_path)
    tube = torch.load(object_tube_path, map_location="cpu", weights_only=False)
    object_rows = _object_chunk_rows(df)
    scene_metric, agg_metric = _metric_lookup()

    fragmentation_rows: list[dict[str, Any]] = []
    for (split, scene), grp in df.groupby(["dataset_split", "scene_id"], dropna=False):
        split = str(split)
        scene = str(scene)
        obj_ids = sorted(str(v) for v in grp["mv_object_id"].dropna().unique())
        by_object = [row for row in object_rows if row["dataset_split"] == split and row["scene_id"] == scene]
        chunk_counts = [int(row["chunk_count"]) for row in by_object]
        pred_scene_object_count = len(obj_ids)
        objects_crossing = sum(1 for count in chunk_counts if count > 1)
        gt_scene_count = int(_num(scene_metric.get((split, scene), {}).get("gt_object_count"), 0))
        pred_per_gt = float(pred_scene_object_count / max(1, gt_scene_count))
        frames_per_object = [int(row["frame_count"]) for row in by_object]
        fragmentation_rate = 1.0 - float(objects_crossing / max(1, pred_scene_object_count))
        scene_row = scene_metric.get((split, scene), {})
        agg_row = agg_metric.get(split, {})
        local_ap_window = _num(agg_row.get("MV_AP_window"))
        raw_scene_ap = _num(scene_row.get("MV_AP_scene"))
        ap_gap = float(local_ap_window - raw_scene_ap)
        fragmentation_dominant = objects_crossing == 0 and pred_scene_object_count > 0 and raw_scene_ap < local_ap_window
        fragmentation_rows.append(
            {
                "schema_version": "stream4d_v100_phase3_fragmentation_row_v1",
                "phase_id": "v100_phase3_scene_fragmentation_audit",
                "dataset_split": split,
                "scene_id": scene,
                "pred_scene_object_count": pred_scene_object_count,
                "gt_scene_object_count": gt_scene_count,
                "pred_per_gt_scene_ratio": pred_per_gt,
                "objects_crossing_multiple_chunks": objects_crossing,
                "mean_chunks_per_scene_object": float(np.mean(chunk_counts)) if chunk_counts else 0.0,
                "max_chunks_per_scene_object": max(chunk_counts) if chunk_counts else 0,
                "mean_pred_frames_per_object": float(np.mean(frames_per_object)) if frames_per_object else 0.0,
                "fragmentation_rate": fragmentation_rate,
                "local_MV_AP_window": local_ap_window,
                "window_fragmented_MV_AP_scene": _num(agg_row.get("MV_AP_scene")),
                "raw_scene_MV_AP_scene": raw_scene_ap,
                "raw_scene_MV_AP50_scene": _num(scene_row.get("MV_AP50_scene")),
                "local_to_scene_MV_AP_gap": ap_gap,
                "same_frame_collision_count": int(_num(agg_row.get("same_frame_collision_count"))),
                "pixel_collision_rate": _num(agg_row.get("pixel_collision_rate")),
                "missing_mask_raster_count": int(_num(agg_row.get("missing_mask_raster_count"))),
                "fragmentation_dominant": fragmentation_dominant,
                "overmerge_dominant": False,
                "local2history_required": objects_crossing == 0,
            }
        )

    all_crossing = sum(int(row["objects_crossing_multiple_chunks"]) for row in fragmentation_rows)
    all_pred = sum(int(row["pred_scene_object_count"]) for row in fragmentation_rows)
    mean_fragmentation = float(np.mean([row["fragmentation_rate"] for row in fragmentation_rows])) if fragmentation_rows else 0.0
    local2history_required = any(bool(row["local2history_required"]) for row in fragmentation_rows)
    fragmentation_confirmed = bool(fragmentation_rows) and all_crossing == 0 and mean_fragmentation >= 0.999
    tube_row_count_ok = int(tube.get("source_row_count", -1)) == int(len(df))
    tube_object_count_ok = len(tube.get("object_ids", [])) == len({str(v) for v in df["mv_object_id"].dropna().unique()})

    gate_rows = [
        {
            "gate_id": "phase2_input_pass",
            "pass": True,
            "expected": "Phase2 pass",
            "observed": phase2.get("decision"),
            "severity": "required",
        },
        {
            "gate_id": "object_tube_matches_parquet",
            "pass": tube_row_count_ok and tube_object_count_ok,
            "expected": "object_tube source_row_count/object_count match parquet",
            "observed": f"tube_rows={tube.get('source_row_count')} parquet_rows={len(df)} tube_objects={len(tube.get('object_ids', []))} parquet_objects={len({str(v) for v in df['mv_object_id'].dropna().unique()})}",
            "severity": "required",
        },
        {
            "gate_id": "fragmentation_source_confirmed",
            "pass": fragmentation_confirmed,
            "expected": "objects_crossing_multiple_chunks=0 and mean fragmentation_rate>=0.999",
            "observed": f"objects_crossing_multiple_chunks={all_crossing}; mean_fragmentation_rate={mean_fragmentation}",
            "severity": "required",
        },
        {
            "gate_id": "local2history_mandatory",
            "pass": local2history_required,
            "expected": "objects_crossing_multiple_chunks=0 implies local2history is mandatory",
            "observed": local2history_required,
            "severity": "routing",
        },
        {
            "gate_id": "overmerge_not_primary",
            "pass": all_crossing == 0,
            "expected": "no existing cross-chunk objects to diagnose as overmerge",
            "observed": f"objects_crossing_multiple_chunks={all_crossing}",
            "severity": "routing",
        },
    ]
    failure_rows = [
        {
            "failure_id": row["gate_id"],
            "severity": row["severity"],
            "expected": row["expected"],
            "observed": row["observed"],
            "repair_direction": (
                "If evaluator/scope mismatch, fix scene adapter before algorithm changes. "
                "If overmerge dominates, add cannot-link before any more merge."
            ),
        }
        for row in gate_rows
        if not bool(row["pass"])
    ]
    phase3_pass = not failure_rows

    frag_csv = OUT_DIR / "fragmentation_rows.csv"
    object_chunk_csv = OUT_DIR / "object_chunk_rows.csv"
    gate_csv = OUT_DIR / "variant_gate_rows.csv"
    failure_csv = OUT_DIR / "variant_failure_rows.csv"
    performance_csv = OUT_DIR / "performance_rows.csv"
    artifact_csv = OUT_DIR / "artifact_manifest_rows.csv"

    _write_csv(frag_csv, fragmentation_rows)
    _write_csv(object_chunk_csv, object_rows)
    _write_csv(gate_csv, gate_rows)
    _write_csv(failure_csv, failure_rows)
    _write_csv(
        performance_csv,
        [
            {
                "schema_version": "stream4d_v100_phase3_performance_row_v1",
                "phase_id": "v100_phase3_scene_fragmentation_audit",
                "case_id": "fragmentation_audit_from_phase2_artifacts",
                "runtime_sec": time.time() - started,
                "parquet_bytes_read": frame_mask_path.stat().st_size,
                "object_tube_bytes_read": object_tube_path.stat().st_size,
                "row_count": len(df),
                "object_count": len(object_rows),
                "gpu_used": False,
            }
        ],
    )
    artifacts = [
        (frag_csv, "csv", "per split/scene fragmentation diagnostics"),
        (object_chunk_csv, "csv", "per object chunk-span diagnostics"),
        (gate_csv, "csv", "phase3 gates"),
        (failure_csv, "csv", "phase3 failures if any"),
        (performance_csv, "csv", "phase3 performance rows"),
    ]
    _write_csv(
        artifact_csv,
        [
            {
                "schema_version": "stream4d_v100_phase3_artifact_manifest_row_v1",
                "phase_id": "v100_phase3_scene_fragmentation_audit",
                "artifact_path": _rel(path),
                "artifact_type": kind,
                "size_bytes": path.stat().st_size if path.exists() else 0,
                "sha256": _sha256(path) if path.exists() and path.is_file() else "",
                "note": note,
            }
            for path, kind, note in artifacts
        ],
    )

    summary = {
        "schema_version": "stream4d_v100_phase3_scene_fragmentation_audit_summary_v1",
        "phase_id": "v100_phase3_scene_fragmentation_audit",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "runtime_sec": time.time() - started,
        "decision": "PASS_FRAGMENTATION_CONFIRMED_ENTER_PHASE4" if phase3_pass else "BLOCK_PHASE4_REPAIR_FRAGMENTATION_DIAGNOSIS",
        "phase3_pass": phase3_pass,
        "failure_count": len(failure_rows),
        "input_phase2_summary": _rel(PHASE2_DIR / "summary.json"),
        "pred_scene_object_count_total": all_pred,
        "objects_crossing_multiple_chunks_total": all_crossing,
        "mean_fragmentation_rate": mean_fragmentation,
        "fragmentation_confirmed": fragmentation_confirmed,
        "local2history_required": local2history_required,
        "overmerge_primary": False,
        "fragmentation_rows": fragmentation_rows,
        "outputs": {
            "summary": _rel(OUT_DIR / "summary.json"),
            "fragmentation_rows": _rel(frag_csv),
            "object_chunk_rows": _rel(object_chunk_csv),
            "variant_gate_rows": _rel(gate_csv),
            "variant_failure_rows": _rel(failure_csv),
            "performance_rows": _rel(performance_csv),
            "artifact_manifest_rows": _rel(artifact_csv),
        },
    }
    _write_json(OUT_DIR / "summary.json", summary)
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))
    return 0 if phase3_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
