#!/usr/bin/env python3
"""GT-only scene stitching oracle for v99 Phase10O primary rows.

This is a diagnostic upper bound. It uses ground-truth object labels to assign
predicted chunk objects to scene-level identities, so it must never be reported
as a method result.
"""

from __future__ import annotations

import csv
import json
import math
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


STREAM3D_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = STREAM3D_ROOT.parent
if str(STREAM3D_ROOT) not in sys.path:
    sys.path.insert(0, str(STREAM3D_ROOT))

from tools import build_v99_phase1_f2_base_reproduction as p1  # noqa: E402
from tools import build_v99_phase10k_holdout_chunk_object_birth_sweep as p10k  # noqa: E402
from tools import build_v99_phase10o_overlap3_scene_stitch_repair as p10o  # noqa: E402


AUDIT_ROOT = STREAM3D_ROOT / "outputs/audit"
OUT_DIR = AUDIT_ROOT / "v99_phase10q_gt_oracle_scene_stitch_diagnostic"
PHASE10O_DIR = AUDIT_ROOT / "v99_phase10o_overlap3_scene_stitch_repair"
BASE_VARIANT = "O0_overlap3_chunk_birth_primary_emit"


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


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _num(value: Any, default: float = 0.0) -> float:
    try:
        if value in ("", None):
            return float(default)
        out = float(value)
    except Exception:
        return float(default)
    return out if math.isfinite(out) else float(default)


def _base_rows() -> list[dict[str, Any]]:
    rows = [dict(row) for row in _read_csv(PHASE10O_DIR / "mv_object_frame_mask_rows.csv") if row.get("variant_id") == BASE_VARIANT]
    if not rows:
        raise RuntimeError("missing Phase10O base rows")
    return rows


def _gt_oracle_assignments(rows: list[dict[str, Any]], eval_scope: dict[str, Any]) -> tuple[dict[str, str], list[dict[str, Any]], dict[str, Any]]:
    by_frame: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_frame[(str(row["scene_id"]), int(float(row["frame_id"])))].append(row)

    object_area: Counter[str] = Counter()
    object_gt_overlap: Counter[tuple[str, int]] = Counter()
    gt_area: Counter[tuple[str, int]] = Counter()
    missing_masks = 0
    for (scene, frame), vals in sorted(by_frame.items()):
        mask_path = eval_scope["mask_path_by_frame"].get((scene, frame))
        if mask_path is None or not mask_path.exists():
            missing_masks += 1
            continue
        label = p1._read_label(mask_path)
        gt = p1._load_gt_2d(scene, frame, tuple(int(v) for v in label.shape[:2]))
        gt_ids, gt_counts = np.unique(gt[gt > 0], return_counts=True)
        for gid, count in zip(gt_ids.tolist(), gt_counts.tolist()):
            gt_area[(scene, int(gid))] += int(count)
        for row in vals:
            oid = str(row["mv_object_id"])
            mask = label == int(float(row["selected_mask_id"]))
            count = int(np.count_nonzero(mask))
            if count <= 0:
                continue
            object_area[oid] += count
            gids, counts = np.unique(gt[mask & (gt > 0)], return_counts=True)
            for gid, overlap in zip(gids.tolist(), counts.tolist()):
                object_gt_overlap[(oid, int(gid))] += int(overlap)

    by_object: dict[str, list[tuple[int, int]]] = defaultdict(list)
    scene_by_object: dict[str, str] = {}
    for row in rows:
        scene_by_object[str(row["mv_object_id"])] = str(row["scene_id"])
    for (oid, gid), overlap in object_gt_overlap.items():
        by_object[oid].append((gid, overlap))

    assignment_rows: list[dict[str, Any]] = []
    mapping: dict[str, str] = {}
    for oid in sorted({str(row["mv_object_id"]) for row in rows}):
        scene = scene_by_object.get(oid, "")
        area = int(object_area.get(oid, 0))
        best_gid = 0
        best_overlap = 0
        if by_object.get(oid):
            best_gid, best_overlap = max(by_object[oid], key=lambda item: (item[1], -item[0]))
        gt_total = int(gt_area.get((scene, best_gid), 0))
        union = area + gt_total - best_overlap
        iou = float(best_overlap / union) if union > 0 else 0.0
        precision = float(best_overlap / area) if area > 0 else 0.0
        recall = float(best_overlap / gt_total) if gt_total > 0 else 0.0
        if best_gid > 0:
            mapping[oid] = f"Q_gt_oracle:{scene}:gt_{best_gid:05d}"
        else:
            mapping[oid] = f"Q_gt_oracle:{oid}"
        assignment_rows.append(
            {
                "schema_version": "stream4d_v99_phase10q_gt_oracle_assignment_v1",
                "phase_id": "v99_phase10q_gt_oracle_scene_stitch_diagnostic",
                "mv_object_id": oid,
                "scene_id": scene,
                "assigned_gt_id": best_gid,
                "object_pixel_area": area,
                "assigned_gt_pixel_area": gt_total,
                "intersection_pixels": int(best_overlap),
                "oracle_iou": iou,
                "oracle_precision": precision,
                "oracle_recall": recall,
                "uses_gt_for_prediction": True,
                "uses_future": False,
            }
        )

    stats = {
        "object_count": len(mapping),
        "assigned_object_count": sum(1 for row in assignment_rows if int(row["assigned_gt_id"]) > 0),
        "missing_mask_frame_count": missing_masks,
        "mean_oracle_iou": float(np.mean([float(row["oracle_iou"]) for row in assignment_rows])) if assignment_rows else 0.0,
        "mean_oracle_precision": float(np.mean([float(row["oracle_precision"]) for row in assignment_rows])) if assignment_rows else 0.0,
        "mean_oracle_recall": float(np.mean([float(row["oracle_recall"]) for row in assignment_rows])) if assignment_rows else 0.0,
    }
    return mapping, assignment_rows, stats


def _apply_mapping(rows: list[dict[str, Any]], *, variant_id: str, mapping: dict[str, str]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        oid = str(row["mv_object_id"])
        new = dict(row)
        new["variant_id"] = variant_id
        new["variant"] = variant_id
        new["phase10q_parent_mv_object_id"] = oid
        new["mv_object_id"] = mapping.get(oid, f"{variant_id}:{oid}")
        new["object_id"] = new["mv_object_id"]
        new["object_id_policy"] = "gt_oracle_scene_identity_diagnostic_not_method"
        new["score_scope"] = "current_chunk_score_gt_oracle_scene_identity"
        new["score_policy"] = str(row.get("score_policy", "")) + "__phase10q_gt_oracle"
        new["uses_gt_for_prediction"] = True
        new["uses_future"] = False
        out.append(new)
    return out


def main() -> int:
    started = datetime.now()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    p10k._patch_phase1_inputs()
    scope = p10o._build_overlap3_scope()
    eval_scope = p10o._eval_scope_from_overlap(scope)
    base_rows = _base_rows()
    mapping, assignment_rows, oracle_stats = _gt_oracle_assignments(base_rows, eval_scope)
    oracle_rows = _apply_mapping(base_rows, variant_id="Q0_gt_oracle_scene_identity", mapping=mapping)
    metric_rows, frame_rows = p1._evaluate_variant("Q0_gt_oracle_scene_identity", oracle_rows, eval_scope)
    aggregate_rows = p1._aggregate_metrics(metric_rows)
    if len(aggregate_rows) != 1:
        raise RuntimeError("expected one aggregate row")
    agg = aggregate_rows[0]
    agg["uses_gt_for_prediction"] = True
    agg["metric_composition"] = "gt_oracle_scene_identity_diagnostic_not_method"

    scene_object_count = len(set(mapping.values()))
    gt_ids_by_scene = defaultdict(set)
    for row in assignment_rows:
        if int(row["assigned_gt_id"]) > 0:
            gt_ids_by_scene[str(row["scene_id"])].add(int(row["assigned_gt_id"]))

    summary = {
        "schema_version": "stream4d_v99_phase10q_gt_oracle_scene_stitch_diagnostic_summary_v1",
        "phase_id": "v99_phase10q_gt_oracle_scene_stitch_diagnostic",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "runtime_sec": (datetime.now() - started).total_seconds(),
        "decision": "DIAGNOSTIC_GT_ORACLE_NOT_METHOD",
        "formal_claim_allowed": False,
        "uses_gt_for_prediction": True,
        "uses_future": False,
        "warning": "GT oracle is an upper-bound diagnostic only and must not be reported as a method result.",
        "oracle_MV_AP_window": float(_num(agg.get("MV_AP_window"))),
        "oracle_MV_AP50_window": float(_num(agg.get("MV_AP50_window"))),
        "oracle_MV_AP_scene": float(_num(agg.get("MV_AP_scene"))),
        "oracle_MV_AP50_scene": float(_num(agg.get("MV_AP50_scene"))),
        "oracle_scene_object_count": scene_object_count,
        "oracle_assigned_gt_object_count": int(sum(len(vals) for vals in gt_ids_by_scene.values())),
        "oracle_stats": oracle_stats,
        "interpretation": (
            "If GT oracle is still far below the F2 scene gate, the blocker is not only local2history stitching; "
            "the local object support/object-birth rows themselves have insufficient scene-level support."
        ),
        "outputs": {
            "summary": _rel(OUT_DIR / "summary.json"),
            "assignment_rows": _rel(OUT_DIR / "gt_oracle_assignment_rows.csv"),
            "variant_metric_rows": _rel(OUT_DIR / "variant_metric_rows.csv"),
            "variant_metric_scene_rows": _rel(OUT_DIR / "variant_metric_scene_rows.csv"),
            "mv_object_frame_mask_rows": _rel(OUT_DIR / "mv_object_frame_mask_rows.csv"),
            "frame_eval_rows": _rel(OUT_DIR / "frame_eval_rows.csv"),
        },
    }
    _write_csv(OUT_DIR / "gt_oracle_assignment_rows.csv", assignment_rows)
    _write_csv(OUT_DIR / "variant_metric_rows.csv", [agg])
    _write_csv(OUT_DIR / "variant_metric_scene_rows.csv", metric_rows)
    _write_csv(OUT_DIR / "mv_object_frame_mask_rows.csv", oracle_rows)
    _write_csv(OUT_DIR / "frame_eval_rows.csv", frame_rows)
    _write_json(OUT_DIR / "summary.json", summary)
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
