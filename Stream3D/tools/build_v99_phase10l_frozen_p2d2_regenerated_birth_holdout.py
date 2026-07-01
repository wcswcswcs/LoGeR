#!/usr/bin/env python3
"""v99 Phase10L frozen P2-D2 score on regenerated holdout object birth.

Phase2 selected the dev-frozen method:
  F2_chunk32_surfel_maskview_birth_thr018 + semantic consistency tie-break.

Earlier holdout projection evaluated the P2-D2 score on fixed legacy F2 rows.
Phase10K showed regenerated holdout chunk object-birth rows pass the local
holdout metric. This script applies the already frozen P2-D2 score policy to
the regenerated holdout thr018 object-birth rows and evaluates the same v65
local-window AP contract.
"""

from __future__ import annotations

import csv
import json
import math
import sys
from collections import defaultdict
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


AUDIT_ROOT = STREAM3D_ROOT / "outputs/audit"
OUT_DIR = AUDIT_ROOT / "v99_phase10l_frozen_p2d2_regenerated_birth_holdout"
PHASE0_DIR = AUDIT_ROOT / "v99_phase0_fact_lock"
PHASE2_DIR = AUDIT_ROOT / "v99_phase2_f2_strengthening"
PHASE10K_DIR = AUDIT_ROOT / "v99_phase10k_holdout_chunk_object_birth_sweep"
FROZEN_DEV_VARIANT = "P2_D2_frame_count_plus_semantic_tiebreak"
FROZEN_BIRTH_VARIANT = "F2_chunk32_surfel_maskview_birth_thr018"
FROZEN_HOLDOUT_VARIANT = "P2_D2_frame_count_plus_semantic_tiebreak__regenerated_chunk_birth_holdout"
EPS = 1e-4


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


def _norm(values: dict[str, float]) -> dict[str, float]:
    vals = [float(v) for v in values.values() if math.isfinite(float(v))]
    if not vals:
        return {key: 0.0 for key in values}
    lo = min(vals)
    hi = max(vals)
    if hi - lo <= 1e-12:
        return {key: 0.0 for key in values}
    return {key: (float(val) - lo) / (hi - lo) for key, val in values.items()}


def _load_holdout_residual_features() -> dict[tuple[str, int, int], np.ndarray]:
    constants = json.loads(p1.SEMANTIC_CONSTANTS.read_text(encoding="utf-8"))
    mu = np.asarray(np.load(p1._project(constants["radio_mu_vector_path"])), dtype=np.float32)
    payload = np.load(p10k.HOLDOUT_RADIO_MASK_FEATURES, allow_pickle=True)
    features = np.asarray(payload["features"], dtype=np.float32)
    residual = p1._normalize_rows(features - mu[None, :])
    out: dict[tuple[str, int, int], np.ndarray] = {}
    for idx in range(residual.shape[0]):
        out[(str(payload["scene_id"][idx]), int(payload["frame_id"][idx]), int(payload["mask_id"][idx]))] = residual[idx]
    return out


def _semantic_norm_by_object(rows: list[dict[str, Any]]) -> dict[str, float]:
    features = _load_holdout_residual_features()
    by_object: dict[str, list[np.ndarray]] = defaultdict(list)
    for row in rows:
        key = (
            str(row["scene_id"]),
            int(row["frame_id"]),
            int(row["selected_mask_id"]),
        )
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


def _make_frozen_rows(parent_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    frames_by_object: dict[str, set[tuple[str, int]]] = defaultdict(set)
    for row in parent_rows:
        frames_by_object[str(row["mv_object_id"])].add((str(row["scene_id"]), int(row["frame_id"])))
    semantic_norm = _semantic_norm_by_object(parent_rows)
    out: list[dict[str, Any]] = []
    for row in parent_rows:
        oid = str(row["mv_object_id"])
        frame_count_score = len(frames_by_object[oid]) / float(p1.CHUNK_SIZE)
        new = dict(row)
        new["variant_id"] = FROZEN_HOLDOUT_VARIANT
        new["variant"] = FROZEN_HOLDOUT_VARIANT
        new["score"] = float(frame_count_score + EPS * semantic_norm.get(oid, 0.0))
        new["score_scope"] = "current_chunk"
        new["score_policy"] = "current_chunk_frame_count_over_32_plus_1e-4_semantic_consistency_tiebreak"
        new["fixed_dev_variant_id"] = FROZEN_DEV_VARIANT
        new["fixed_birth_variant_id"] = FROZEN_BIRTH_VARIANT
        new["phase10l_parent_variant_id"] = row.get("variant_id", "")
        new["phase10l_frame_count_score"] = frame_count_score
        new["phase10l_semantic_norm"] = semantic_norm.get(oid, 0.0)
        new["uses_gt_for_prediction"] = False
        new["uses_future"] = False
        return_row = new
        out.append(return_row)
    return out


def _dev_variant_row() -> dict[str, Any]:
    for row in _read_csv(PHASE2_DIR / "variant_metric_rows.csv"):
        if row.get("variant_id") == FROZEN_DEV_VARIANT:
            return dict(row)
    raise RuntimeError(f"missing dev metric row for {FROZEN_DEV_VARIANT}")


def main() -> int:
    started = datetime.now()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    p10k._patch_phase1_inputs()
    phase0 = json.loads((PHASE0_DIR / "summary.json").read_text(encoding="utf-8"))
    scope = p1._load_source_scope()
    parent_rows = [
        dict(row)
        for row in _read_csv(PHASE10K_DIR / "mv_object_frame_mask_rows.csv")
        if row.get("variant_id") == FROZEN_BIRTH_VARIANT
    ]
    if not parent_rows:
        raise RuntimeError(f"missing Phase10K rows for {FROZEN_BIRTH_VARIANT}")
    frozen_rows = _make_frozen_rows(parent_rows)
    metric_rows, frame_rows = p1._evaluate_variant(FROZEN_HOLDOUT_VARIANT, frozen_rows, scope)
    aggregate_rows = p1._aggregate_metrics(metric_rows)
    if not aggregate_rows:
        raise RuntimeError("no aggregate metric row")
    agg = aggregate_rows[0]
    dev = _dev_variant_row()

    dev_gate_window = float(phase0["F2_base_full_dev_MV_AP_window"]) + 0.005
    dev_gate_ap50 = float(phase0["F2_base_full_dev_MV_AP50_window"]) + 0.010
    hold_gate_window = float(phase0["F2_base_holdout_MV_AP_window"]) + 0.005
    hold_gate_ap50 = float(phase0["F2_base_holdout_MV_AP50_window"]) + 0.010
    dev_gate = _num(dev["MV_AP_window"]) >= dev_gate_window and _num(dev["MV_AP50_window"]) >= dev_gate_ap50
    hold_gate = _num(agg["MV_AP_window"]) >= hold_gate_window and _num(agg["MV_AP50_window"]) >= hold_gate_ap50
    safety_gate = (
        int(_num(agg.get("same_frame_collision_count"), 1)) == 0
        and int(_num(agg.get("missing_mask_raster_count"), 1)) == 0
        and not bool(scope.get("source_uses_future", False))
        and not bool(scope.get("source_uses_gt_for_prediction", False))
    )
    metric_gate_pass = bool(dev_gate and hold_gate and safety_gate)

    paired = [
        {
            "schema_version": "stream4d_v99_phase10l_frozen_p2d2_metric_v1",
            "phase_id": "v99_phase10l_frozen_p2d2_regenerated_birth_holdout",
            "fixed_dev_variant_id": FROZEN_DEV_VARIANT,
            "holdout_variant_id": FROZEN_HOLDOUT_VARIANT,
            "dev_MV_AP_window": dev["MV_AP_window"],
            "dev_MV_AP50_window": dev["MV_AP50_window"],
            "dev_MV_AP_scene": dev["MV_AP_scene"],
            "dev_MV_AP50_scene": dev["MV_AP50_scene"],
            "dev_gate_pass": dev_gate,
            "holdout_MV_AP_window": agg["MV_AP_window"],
            "holdout_MV_AP50_window": agg["MV_AP50_window"],
            "holdout_MV_AP25_window": agg["MV_AP25_window"],
            "holdout_MV_AP_scene": agg["MV_AP_scene"],
            "holdout_MV_AP50_scene": agg["MV_AP50_scene"],
            "holdout_gate_pass": hold_gate,
            "metric_gate_pass": metric_gate_pass,
            "formal_claim_allowed": False,
            "uses_gt_for_prediction": False,
            "uses_future": False,
        }
    ]
    gate_rows = [
        {
            "gate_id": "dev_frozen_p2d2_gate",
            "pass": dev_gate,
            "expected": f"MV_AP_window>={dev_gate_window} and MV_AP50_window>={dev_gate_ap50}",
            "observed": f"MV_AP_window={dev['MV_AP_window']} MV_AP50_window={dev['MV_AP50_window']}",
            "severity": "method_gate",
        },
        {
            "gate_id": "regenerated_birth_holdout_gate",
            "pass": hold_gate,
            "expected": f"MV_AP_window>={hold_gate_window} and MV_AP50_window>={hold_gate_ap50}",
            "observed": f"MV_AP_window={agg['MV_AP_window']} MV_AP50_window={agg['MV_AP50_window']}",
            "severity": "method_gate",
        },
        {
            "gate_id": "safety_gate",
            "pass": safety_gate,
            "expected": "same_frame_collision=0 missing_mask=0 uses_future=false uses_gt=false",
            "observed": f"same_frame_collision={agg.get('same_frame_collision_count')} missing_mask={agg.get('missing_mask_raster_count')} uses_future={scope.get('source_uses_future')} uses_gt={scope.get('source_uses_gt_for_prediction')}",
            "severity": "safety_gate",
        },
        {
            "gate_id": "formal_claim_allowed_after_repaired_projection",
            "pass": False,
            "expected": "reviewer accepts Phase10L as missing holdout-projection repair with frozen dev method",
            "observed": "post-final repair run; surfel identity chunk-causal proof remains waived/not formalized",
            "severity": "formal_claim_review",
        },
    ]
    failure_rows = [
        {
            "failure_id": row["gate_id"],
            "severity": row["severity"],
            "expected": row["expected"],
            "observed": row["observed"],
            "repair_direction": "Formalize regenerated holdout projection protocol and chunk-causal surfel identity proof before paper claim.",
        }
        for row in gate_rows
        if not bool(row["pass"])
    ]
    summary = {
        "schema_version": "stream4d_v99_phase10l_frozen_p2d2_regenerated_birth_holdout_summary_v1",
        "phase_id": "v99_phase10l_frozen_p2d2_regenerated_birth_holdout",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "runtime_sec": (datetime.now() - started).total_seconds(),
        "decision": "GO_METRIC_REPAIRED_HOLDOUT_PROJECTION_REQUIRES_FORMAL_REVIEW" if metric_gate_pass else "NO_GO_FROZEN_P2D2_REGENERATED_BIRTH_HOLDOUT",
        "metric_gate_pass": metric_gate_pass,
        "formal_claim_allowed": False,
        "uses_gt_for_prediction": False,
        "uses_future": False,
        "fixed_dev_variant_id": FROZEN_DEV_VARIANT,
        "fixed_birth_variant_id": FROZEN_BIRTH_VARIANT,
        "holdout_variant_id": FROZEN_HOLDOUT_VARIANT,
        "dev_MV_AP_window": float(_num(dev["MV_AP_window"])),
        "dev_MV_AP50_window": float(_num(dev["MV_AP50_window"])),
        "dev_MV_AP_scene": float(_num(dev["MV_AP_scene"])),
        "dev_MV_AP50_scene": float(_num(dev["MV_AP50_scene"])),
        "holdout_MV_AP_window": float(_num(agg["MV_AP_window"])),
        "holdout_MV_AP50_window": float(_num(agg["MV_AP50_window"])),
        "holdout_MV_AP_scene": float(_num(agg["MV_AP_scene"])),
        "holdout_MV_AP50_scene": float(_num(agg["MV_AP50_scene"])),
        "dev_gate_pass": dev_gate,
        "holdout_gate_pass": hold_gate,
        "safety_gate_pass": safety_gate,
        "formal_blocker": "post-final repaired projection plus surfel identity chunk-causal proof not formalized",
        "outputs": {
            "summary": _rel(OUT_DIR / "summary.json"),
            "paired_metric_rows": _rel(OUT_DIR / "paired_metric_rows.csv"),
            "holdout_metric_rows": _rel(OUT_DIR / "holdout_metric_rows.csv"),
            "holdout_metric_aggregate_rows": _rel(OUT_DIR / "holdout_metric_aggregate_rows.csv"),
            "variant_gate_rows": _rel(OUT_DIR / "variant_gate_rows.csv"),
            "variant_failure_rows": _rel(OUT_DIR / "variant_failure_rows.csv"),
            "mv_object_frame_mask_rows": _rel(OUT_DIR / "mv_object_frame_mask_rows.csv"),
        },
    }

    _write_csv(OUT_DIR / "paired_metric_rows.csv", paired)
    _write_csv(OUT_DIR / "holdout_metric_rows.csv", metric_rows)
    _write_csv(OUT_DIR / "holdout_metric_aggregate_rows.csv", aggregate_rows)
    _write_csv(OUT_DIR / "frame_eval_rows.csv", frame_rows)
    _write_csv(OUT_DIR / "mv_object_frame_mask_rows.csv", frozen_rows)
    _write_csv(OUT_DIR / "variant_gate_rows.csv", gate_rows)
    _write_csv(OUT_DIR / "variant_failure_rows.csv", failure_rows)
    _write_json(OUT_DIR / "summary.json", summary)
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))
    return 0 if metric_gate_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
