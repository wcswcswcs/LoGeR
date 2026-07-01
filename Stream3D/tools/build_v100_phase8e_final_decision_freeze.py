#!/usr/bin/env python3
"""Final v100 decision freeze after DA3 surface-component split repair."""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any


STREAM3D_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = STREAM3D_ROOT.parent
if str(STREAM3D_ROOT) not in sys.path:
    sys.path.insert(0, str(STREAM3D_ROOT))

from tools import build_v100_phase4h_overlap3_exact_history_memory as p4h  # noqa: E402


AUDIT_ROOT = STREAM3D_ROOT / "outputs/audit"
OUT_DIR = AUDIT_ROOT / "v100_phase8e_final_decision_freeze"
PHASE2C = AUDIT_ROOT / "v100_phase2c_overlap3_local_repair"
PHASE4R = AUDIT_ROOT / "v100_phase4r_position_union_repair"
PHASE5C = AUDIT_ROOT / "v100_phase5c_da3_broad_split_repair"
PHASE5D = AUDIT_ROOT / "v100_phase5d_da3_surface_component_split_repair"
PHASE8D = AUDIT_ROOT / "v100_phase8d_continued_repair_decision_freeze"


def _rel(path: Path | str) -> str:
    return p4h._rel(path)


def _num(value: Any, default: float = 0.0) -> float:
    return p4h._num(value, default)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    p4h._write_csv(path, rows)


def _write_json(path: Path, payload: Any) -> None:
    p4h._write_json(path, payload)


def _artifact_rows(paths: list[tuple[Path, str, str]]) -> list[dict[str, Any]]:
    return [
        {
            "schema_version": "stream4d_v100_phase8e_artifact_manifest_row_v1",
            "phase_id": "v100_phase8e_final_decision_freeze",
            "artifact_path": _rel(path),
            "artifact_type": kind,
            "size_bytes": path.stat().st_size if path.exists() else 0,
            "sha256": p4h._sha256(path) if path.exists() and path.is_file() else "",
            "note": note,
        }
        for path, kind, note in paths
    ]


def main() -> int:
    started = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    phase2c = _load_json(PHASE2C / "summary.json")
    phase4r = _load_json(PHASE4R / "summary.json")
    phase5c = _load_json(PHASE5C / "summary.json")
    phase5d = _load_json(PHASE5D / "summary.json")
    phase8d = _load_json(PHASE8D / "summary.json")
    baselines = p4h._phase0_baselines()
    f2_dev = baselines["F2_base_full_dev"]
    f2_holdout = baselines["F2_base_holdout"]
    dev_scene_gate = _num(f2_dev["MV_AP_scene"]) + 0.010
    dev_ap50_gate = _num(f2_dev["MV_AP50_scene"]) + 0.015
    hold_scene_gate = _num(f2_holdout["MV_AP_scene"]) + 0.006
    hold_ap50_gate = _num(f2_holdout["MV_AP50_scene"]) + 0.010

    local_pass = bool(phase2c.get("phase2c_pass")) and bool(phase2c.get("formal_claim_allowed"))
    best_scene = {
        "source_id": "phase4r_position_union",
        "variant_id": phase4r.get("best_variant_id"),
        "dev_MV_AP_scene": phase4r.get("best_dev_MV_AP_scene"),
        "dev_MV_AP50_scene": phase4r.get("best_dev_MV_AP50_scene"),
        "holdout_MV_AP_scene": phase4r.get("best_holdout_MV_AP_scene"),
        "holdout_MV_AP50_scene": phase4r.get("best_holdout_MV_AP50_scene"),
        "dev_scene_gate": dev_scene_gate,
        "dev_ap50_gate": dev_ap50_gate,
        "holdout_scene_gate": hold_scene_gate,
        "holdout_ap50_gate": hold_ap50_gate,
        "artifact": _rel(PHASE4R / "summary.json"),
    }
    scene_pass = bool(
        _num(best_scene["dev_MV_AP_scene"]) >= dev_scene_gate
        and _num(best_scene["dev_MV_AP50_scene"]) >= dev_ap50_gate
        and _num(best_scene["holdout_MV_AP_scene"]) >= hold_scene_gate
        and _num(best_scene["holdout_MV_AP50_scene"]) >= hold_ap50_gate
    )
    full_goal_achieved = bool(local_pass and scene_pass)
    decision = (
        "GO_LOCAL_AND_SCENE_PHASE2C_OVERLAP3"
        if full_goal_achieved
        else "GO_LOCAL_ONLY_PHASE2C_OVERLAP3__FINAL_NO_GO_SCENE_STITCHING_AFTER_PHASE5D"
        if local_pass
        else "NO_GO_CHUNK_CAUSAL_LOCAL"
    )

    decision_rows = [
        {
            "schema_version": "stream4d_v100_phase8e_decision_row_v1",
            "phase_id": "v100_phase8e_final_decision_freeze",
            "decision": decision,
            "previous_decision": phase8d.get("decision"),
            "full_goal_achieved": full_goal_achieved,
            "local_claim_allowed": local_pass,
            "scene_claim_allowed": full_goal_achieved,
            "uses_gt_for_prediction": False,
        }
    ]
    metric_rows = [
        {
            "schema_version": "stream4d_v100_phase8e_metric_row_v1",
            "phase_id": "v100_phase8e_final_decision_freeze",
            "row_id": "phase2c_local_claim",
            "dev_MV_AP_window": phase2c.get("dev_MV_AP_window"),
            "dev_MV_AP50_window": phase2c.get("dev_MV_AP50_window"),
            "holdout_MV_AP_window": phase2c.get("holdout_MV_AP_window"),
            "holdout_MV_AP50_window": phase2c.get("holdout_MV_AP50_window"),
            "formal_claim_allowed": phase2c.get("formal_claim_allowed"),
        },
        {
            "schema_version": "stream4d_v100_phase8e_metric_row_v1",
            "phase_id": "v100_phase8e_final_decision_freeze",
            "row_id": "best_scene_attempt_phase4r",
            **{k: v for k, v in best_scene.items() if k != "artifact"},
            "formal_claim_allowed": False,
        },
        {
            "schema_version": "stream4d_v100_phase8e_metric_row_v1",
            "phase_id": "v100_phase8e_final_decision_freeze",
            "row_id": "phase5c_da3_median_split",
            "decision": phase5c.get("decision"),
            "best_variant_id": phase5c.get("best_variant_id"),
            "best_split_variant_id": phase5c.get("best_split_variant_id"),
            "best_split_holdout_MV_AP_scene": phase5c.get("best_split_holdout_MV_AP_scene"),
            "split_component_row_count": phase5c.get("split_component_row_count"),
            "formal_claim_allowed": False,
        },
        {
            "schema_version": "stream4d_v100_phase8e_metric_row_v1",
            "phase_id": "v100_phase8e_final_decision_freeze",
            "row_id": "phase5d_da3_surface_component_split",
            "decision": phase5d.get("decision"),
            "best_variant_id": phase5d.get("best_variant_id"),
            "best_split_variant_id": phase5d.get("best_split_variant_id"),
            "best_split_holdout_MV_AP_scene": phase5d.get("best_split_holdout_MV_AP_scene"),
            "split_component_row_count": phase5d.get("split_component_row_count"),
            "formal_claim_allowed": False,
        },
    ]
    blocker_rows = [
        {
            "schema_version": "stream4d_v100_phase8e_blocker_row_v1",
            "phase_id": "v100_phase8e_final_decision_freeze",
            "blocker_id": "scene_local2history_not_solved",
            "status": "open",
            "evidence": f"best_scene_holdout={best_scene['holdout_MV_AP_scene']} gate={hold_scene_gate}; best_scene_holdout_ap50={best_scene['holdout_MV_AP50_scene']} gate={hold_ap50_gate}",
            "conclusion": "Full v100 target is not achieved; do not claim scene/local2history success.",
        },
        {
            "schema_version": "stream4d_v100_phase8e_blocker_row_v1",
            "phase_id": "v100_phase8e_final_decision_freeze",
            "blocker_id": "da3_split_family_closed",
            "status": "closed_as_no_go_for_split_heuristics",
            "evidence": f"phase5c_best={phase5c.get('best_variant_id')} phase5d_best={phase5d.get('best_variant_id')} phase5d_best_split_holdout_scene={phase5d.get('best_split_holdout_MV_AP_scene')}",
            "conclusion": "Both median-depth and connected surface-component DA3 splits failed to beat no-split baseline; further threshold sweeps are not justified by this evidence.",
        },
        {
            "schema_version": "stream4d_v100_phase8e_blocker_row_v1",
            "phase_id": "v100_phase8e_final_decision_freeze",
            "blocker_id": "missing_cross_chunk_identity_witness",
            "status": "open",
            "evidence": "Phase4r small additive gain remains far below holdout gates; Phase5c/5d per-frame split variants do not solve cross-chunk identity.",
            "conclusion": "The remaining blocker is a stronger non-GT cross-chunk identity witness, not per-frame DA3 mask splitting.",
        },
    ]

    decision_csv = OUT_DIR / "decision_rows.csv"
    metric_csv = OUT_DIR / "metric_rows.csv"
    blocker_csv = OUT_DIR / "blocker_rows.csv"
    artifact_csv = OUT_DIR / "artifact_manifest_rows.csv"
    summary_json = OUT_DIR / "summary.json"
    _write_csv(decision_csv, decision_rows)
    _write_csv(metric_csv, metric_rows)
    _write_csv(blocker_csv, blocker_rows)
    _write_csv(
        artifact_csv,
        _artifact_rows(
            [
                (decision_csv, "csv", "Phase8e decision rows"),
                (metric_csv, "csv", "Final metric evidence"),
                (blocker_csv, "csv", "Final blocker rows"),
            ]
        ),
    )
    summary = {
        "schema_version": "stream4d_v100_phase8e_final_decision_freeze_summary_v1",
        "phase_id": "v100_phase8e_final_decision_freeze",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "runtime_sec": time.time() - started,
        "decision": decision,
        "previous_decision": phase8d.get("decision"),
        "full_goal_achieved": full_goal_achieved,
        "local_claim_allowed": local_pass,
        "scene_claim_allowed": full_goal_achieved,
        "best_current_scene_attempt": best_scene,
        "phase2c_overlap3_local": {
            "dev_MV_AP_window": phase2c.get("dev_MV_AP_window"),
            "dev_MV_AP50_window": phase2c.get("dev_MV_AP50_window"),
            "holdout_MV_AP_window": phase2c.get("holdout_MV_AP_window"),
            "holdout_MV_AP50_window": phase2c.get("holdout_MV_AP50_window"),
            "formal_claim_allowed": phase2c.get("formal_claim_allowed"),
        },
        "phase5d_da3_surface_component_split": {
            "decision": phase5d.get("decision"),
            "best_variant_id": phase5d.get("best_variant_id"),
            "best_split_variant_id": phase5d.get("best_split_variant_id"),
            "best_split_holdout_MV_AP_scene": phase5d.get("best_split_holdout_MV_AP_scene"),
            "split_component_row_count": phase5d.get("split_component_row_count"),
        },
        "uses_gt_for_prediction": False,
        "outputs": {
            "summary": _rel(summary_json),
            "decision_rows": _rel(decision_csv),
            "metric_rows": _rel(metric_csv),
            "blocker_rows": _rel(blocker_csv),
            "artifact_manifest_rows": _rel(artifact_csv),
        },
    }
    _write_json(summary_json, summary)
    print(json.dumps(p4h._jsonable(summary), indent=2, sort_keys=True))
    return 0 if full_goal_achieved else 2


if __name__ == "__main__":
    raise SystemExit(main())
