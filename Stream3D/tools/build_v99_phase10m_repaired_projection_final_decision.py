#!/usr/bin/env python3
"""v99 Phase10M final decision after repaired regenerated holdout projection."""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


STREAM3D_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = STREAM3D_ROOT.parent
AUDIT_ROOT = STREAM3D_ROOT / "outputs/audit"
OUT_DIR = AUDIT_ROOT / "v99_phase10m_repaired_projection_final_decision"
PHASE0 = AUDIT_ROOT / "v99_phase0_fact_lock/summary.json"
PHASE9 = AUDIT_ROOT / "v99_phase9_scene_local2history/summary.json"
PHASE10L = AUDIT_ROOT / "v99_phase10l_frozen_p2d2_regenerated_birth_holdout/summary.json"
OLD_FINAL = AUDIT_ROOT / "v99_phase10_holdout_final_decision/summary.json"


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


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    started = datetime.now()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    phase0 = _load(PHASE0)
    phase9 = _load(PHASE9)
    phase10l = _load(PHASE10L)
    old_final = _load(OLD_FINAL) if OLD_FINAL.exists() else {}

    local_gate_pass = bool(phase10l.get("metric_gate_pass")) and bool(phase10l.get("holdout_gate_pass"))
    scene_gate_pass = bool(phase9.get("scene_gate_pass"))
    formal_claim_allowed = False
    decision = (
        "GO_METRIC_LOCAL_REPAIRED_PROJECTION_SCENE_NO_GO_FORMAL_REVIEW_REQUIRED"
        if local_gate_pass and not scene_gate_pass
        else "NO_GO_REPAIRED_PROJECTION_FINAL"
    )
    gate_rows = [
        {
            "gate_id": "local_mv_ap_window_gate",
            "pass": local_gate_pass,
            "expected": {
                "dev_MV_AP_window": float(phase0["F2_base_full_dev_MV_AP_window"]) + 0.005,
                "dev_MV_AP50_window": float(phase0["F2_base_full_dev_MV_AP50_window"]) + 0.010,
                "holdout_MV_AP_window": float(phase0["F2_base_holdout_MV_AP_window"]) + 0.005,
                "holdout_MV_AP50_window": float(phase0["F2_base_holdout_MV_AP50_window"]) + 0.010,
            },
            "observed": {
                "dev_MV_AP_window": phase10l["dev_MV_AP_window"],
                "dev_MV_AP50_window": phase10l["dev_MV_AP50_window"],
                "holdout_MV_AP_window": phase10l["holdout_MV_AP_window"],
                "holdout_MV_AP50_window": phase10l["holdout_MV_AP50_window"],
            },
            "severity": "method_gate",
        },
        {
            "gate_id": "scene_mv_ap_scene_gate",
            "pass": scene_gate_pass,
            "expected": {
                "MV_AP_scene": float(phase0["F2_base_full_dev_MV_AP_scene"]) + 0.010,
                "MV_AP50_scene": float(phase0["F2_base_full_dev_MV_AP50_scene"]) + 0.015,
            },
            "observed": {
                "phase9_best_scene_variant": phase9.get("best_scene_variant"),
                "phase9_best_scene_MV_AP_scene": phase9.get("best_scene_MV_AP_scene"),
                "phase9_best_scene_MV_AP50_scene": phase9.get("best_scene_MV_AP50_scene"),
                "phase10l_holdout_MV_AP_scene": phase10l.get("holdout_MV_AP_scene"),
                "phase10l_holdout_MV_AP50_scene": phase10l.get("holdout_MV_AP50_scene"),
            },
            "severity": "scene_gate",
        },
        {
            "gate_id": "formal_claim_allowed",
            "pass": formal_claim_allowed,
            "expected": "fresh/repaired projection accepted plus chunk-causal surfel identity proof formalized",
            "observed": phase10l.get("formal_blocker"),
            "severity": "formal_claim_review",
        },
    ]
    failure_rows = [
        {
            "failure_id": row["gate_id"],
            "severity": row["severity"],
            "expected": row["expected"],
            "observed": row["observed"],
            "repair_direction": (
                "Scene/local2history needs a separate stitching method."
                if row["gate_id"] == "scene_mv_ap_scene_gate"
                else "Formalize repaired projection and chunk-causal surfel identity proof."
            ),
        }
        for row in gate_rows
        if not bool(row["pass"])
    ]
    method_rows = [
        {
            "schema_version": "stream4d_v99_phase10m_method_summary_v1",
            "method_id": "F2_chunk32_surfel_maskview_birth_thr018_plus_P2_D2_semantic_tiebreak_regenerated_holdout",
            "fixed_birth_variant_id": phase10l["fixed_birth_variant_id"],
            "fixed_dev_variant_id": phase10l["fixed_dev_variant_id"],
            "holdout_variant_id": phase10l["holdout_variant_id"],
            "dev_MV_AP_window": phase10l["dev_MV_AP_window"],
            "dev_MV_AP50_window": phase10l["dev_MV_AP50_window"],
            "dev_MV_AP_scene": phase10l["dev_MV_AP_scene"],
            "dev_MV_AP50_scene": phase10l["dev_MV_AP50_scene"],
            "holdout_MV_AP_window": phase10l["holdout_MV_AP_window"],
            "holdout_MV_AP50_window": phase10l["holdout_MV_AP50_window"],
            "holdout_MV_AP_scene": phase10l["holdout_MV_AP_scene"],
            "holdout_MV_AP50_scene": phase10l["holdout_MV_AP50_scene"],
            "local_metric_gate_pass": local_gate_pass,
            "scene_gate_pass": scene_gate_pass,
            "formal_claim_allowed": formal_claim_allowed,
        }
    ]
    summary = {
        "schema_version": "stream4d_v99_phase10m_repaired_projection_final_decision_summary_v1",
        "phase_id": "v99_phase10m_repaired_projection_final_decision",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "runtime_sec": (datetime.now() - started).total_seconds(),
        "decision": decision,
        "local_metric_gate_pass": local_gate_pass,
        "scene_gate_pass": scene_gate_pass,
        "formal_claim_allowed": formal_claim_allowed,
        "method_label": "F2 regenerated chunk object birth + P2_D2 semantic tie-break",
        "old_final_decision": old_final.get("decision", ""),
        "old_final_summary": _rel(OLD_FINAL),
        "phase10l_summary": _rel(PHASE10L),
        "phase9_summary": _rel(PHASE9),
        "claim_scope": {
            "allowed_now": "local MV_AP_window/MV_AP50_window metric evidence, with formal review caveat",
            "not_allowed_now": "scene/local2history success; paper-final claim without repaired-projection review",
        },
        "outputs": {
            "summary": _rel(OUT_DIR / "summary.json"),
            "method_rows": _rel(OUT_DIR / "method_rows.csv"),
            "gate_rows": _rel(OUT_DIR / "gate_rows.csv"),
            "failure_rows": _rel(OUT_DIR / "failure_rows.csv"),
        },
    }
    _write_csv(OUT_DIR / "method_rows.csv", method_rows)
    _write_csv(OUT_DIR / "gate_rows.csv", gate_rows)
    _write_csv(OUT_DIR / "failure_rows.csv", failure_rows)
    _write_json(OUT_DIR / "summary.json", summary)
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))
    return 0 if local_gate_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
