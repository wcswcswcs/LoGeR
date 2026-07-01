#!/usr/bin/env python3
"""Audit why Phase10L repaired local method has very low MV_AP_scene."""

from __future__ import annotations

import csv
import json
import statistics
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


STREAM3D_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = STREAM3D_ROOT.parent
AUDIT_ROOT = STREAM3D_ROOT / "outputs/audit"
OUT_DIR = AUDIT_ROOT / "v99_phase10n_scene_fragmentation_audit"
PHASE10L_DIR = AUDIT_ROOT / "v99_phase10l_frozen_p2d2_regenerated_birth_holdout"


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


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(data), indent=2, sort_keys=True) + "\n", encoding="utf-8")


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


def _num(value: Any, default: float = 0.0) -> float:
    try:
        if value in ("", None):
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def _median(values: list[int]) -> float:
    return float(statistics.median(values)) if values else 0.0


def main() -> int:
    started = datetime.now()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    object_rows = _read_csv(PHASE10L_DIR / "mv_object_frame_mask_rows.csv")
    metric_rows = _read_csv(PHASE10L_DIR / "holdout_metric_rows.csv")
    summary = json.loads((PHASE10L_DIR / "summary.json").read_text(encoding="utf-8"))

    frames_by_scene: dict[str, set[int]] = defaultdict(set)
    objects_by_scene: dict[str, set[str]] = defaultdict(set)
    objects_by_scene_chunk: dict[tuple[str, str], set[str]] = defaultdict(set)
    frames_by_object: dict[tuple[str, str], set[int]] = defaultdict(set)
    chunks_by_object: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in object_rows:
        scene = str(row["scene_id"])
        chunk = str(row["chunk_id"])
        oid = str(row["mv_object_id"])
        frame = int(float(row["frame_id"]))
        frames_by_scene[scene].add(frame)
        objects_by_scene[scene].add(oid)
        objects_by_scene_chunk[(scene, chunk)].add(oid)
        frames_by_object[(scene, oid)].add(frame)
        chunks_by_object[(scene, oid)].add(chunk)

    metric_by_scene_scope = {(row["scene_id"], row["metric_scope"]): row for row in metric_rows}
    scene_rows: list[dict[str, Any]] = []
    for scene in sorted(objects_by_scene):
        frame_counts = [len(frames) for (row_scene, _oid), frames in frames_by_object.items() if row_scene == scene]
        chunk_counts = [len(chunks) for (row_scene, _oid), chunks in chunks_by_object.items() if row_scene == scene]
        local = metric_by_scene_scope.get((scene, "local_window_gt_projection_chunk32"), {})
        scene_metric = metric_by_scene_scope.get((scene, "scene_level_raw_gt_chunk_fragmented_or_legacy_id"), {})
        gt_scene = int(_num(scene_metric.get("gt_object_count")))
        pred_scene = int(_num(scene_metric.get("pred_object_count")))
        scene_rows.append(
            {
                "schema_version": "stream4d_v99_phase10n_scene_fragmentation_scene_v1",
                "phase_id": "v99_phase10n_scene_fragmentation_audit",
                "scene_id": scene,
                "frame_count": len(frames_by_scene[scene]),
                "pred_scene_object_count": pred_scene,
                "gt_scene_object_count": gt_scene,
                "pred_per_gt_scene_ratio": pred_scene / max(1, gt_scene),
                "chunk_count": len({chunk for row_scene, chunk in objects_by_scene_chunk if row_scene == scene}),
                "mean_pred_frames_per_object": float(np.mean(frame_counts)) if frame_counts else 0.0,
                "median_pred_frames_per_object": _median(frame_counts),
                "max_pred_frames_per_object": max(frame_counts) if frame_counts else 0,
                "max_chunks_per_pred_object": max(chunk_counts) if chunk_counts else 0,
                "objects_crossing_multiple_chunks": sum(1 for value in chunk_counts if value > 1),
                "object_ids_are_chunk_scoped": all(":c" in oid for oid in objects_by_scene[scene]),
                "local_MV_AP_window": local.get("MV_AP_window", ""),
                "local_MV_AP50_window": local.get("MV_AP50_window", ""),
                "scene_MV_AP_scene": scene_metric.get("MV_AP_scene", ""),
                "scene_MV_AP50_scene": scene_metric.get("MV_AP50_scene", ""),
                "scene_MV_AP25_scene": scene_metric.get("MV_AP25_scene", ""),
                "scene_ScoreFreeMatch50_scene": scene_metric.get("ScoreFreeMatch50_scene", ""),
            }
        )

    low_scene_expected_from_fragmentation = all(
        bool(row["object_ids_are_chunk_scoped"])
        and int(row["objects_crossing_multiple_chunks"]) == 0
        and float(row["pred_per_gt_scene_ratio"]) > 3.0
        for row in scene_rows
    )
    out_summary = {
        "schema_version": "stream4d_v99_phase10n_scene_fragmentation_audit_summary_v1",
        "phase_id": "v99_phase10n_scene_fragmentation_audit",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "runtime_sec": (datetime.now() - started).total_seconds(),
        "decision": "SCENE_LOW_EXPECTED_FROM_CHUNK_FRAGMENTATION" if low_scene_expected_from_fragmentation else "SCENE_LOW_NEEDS_FURTHER_AUDIT",
        "phase10l_summary": _rel(PHASE10L_DIR / "summary.json"),
        "phase10l_holdout_MV_AP_window": summary.get("holdout_MV_AP_window"),
        "phase10l_holdout_MV_AP50_window": summary.get("holdout_MV_AP50_window"),
        "phase10l_holdout_MV_AP_scene": summary.get("holdout_MV_AP_scene"),
        "phase10l_holdout_MV_AP50_scene": summary.get("holdout_MV_AP50_scene"),
        "scene_rows": scene_rows,
        "evaluator_contract": {
            "local": "local_window_gt_projection_chunk32 uses window-scoped GT ids",
            "scene": "scene_level_raw_gt_chunk_fragmented_or_legacy_id uses raw scene GT ids with current chunk-scoped prediction ids",
        },
        "outputs": {
            "summary": _rel(OUT_DIR / "summary.json"),
            "scene_rows": _rel(OUT_DIR / "scene_rows.csv"),
        },
    }
    _write_csv(OUT_DIR / "scene_rows.csv", scene_rows)
    _write_json(OUT_DIR / "summary.json", out_summary)
    print(json.dumps(_jsonable(out_summary), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
