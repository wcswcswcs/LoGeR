#!/usr/bin/env python3
"""Freeze v100 decision after Phase4m-Phase5b continued repairs."""

from __future__ import annotations

import csv
import json
import sys
import time
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


AUDIT_ROOT = STREAM3D_ROOT / "outputs/audit"
OUT_DIR = AUDIT_ROOT / "v100_phase8c_continued_repair_decision_freeze"
PHASE2C = AUDIT_ROOT / "v100_phase2c_overlap3_local_repair"
PHASE4M = AUDIT_ROOT / "v100_phase4m_temporal_maskview_history_memory"
PHASE4N = AUDIT_ROOT / "v100_phase4n_gt_link_precision_diagnostic"
PHASE4O = AUDIT_ROOT / "v100_phase4o_union_history_evidence_repair"
PHASE4P = AUDIT_ROOT / "v100_phase4p_multi_semantic_union_repair"
PHASE5B = AUDIT_ROOT / "v100_phase5b_current_union_d4rt_support_audit"
PHASE8B = AUDIT_ROOT / "v100_phase8b_continued_decision_freeze"


def _rel(path: Path | str) -> str:
    return p4h._rel(path)


def _num(value: Any, default: float = 0.0) -> float:
    return p4h._num(value, default)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    p4h._write_csv(path, rows)


def _write_json(path: Path, payload: Any) -> None:
    p4h._write_json(path, payload)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _artifact_rows(paths: list[tuple[Path, str, str]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path, kind, note in paths:
        rows.append(
            {
                "schema_version": "stream4d_v100_phase8c_artifact_manifest_row_v1",
                "phase_id": "v100_phase8c_continued_repair_decision_freeze",
                "artifact_path": _rel(path),
                "artifact_type": kind,
                "size_bytes": path.stat().st_size if path.exists() else 0,
                "sha256": p4h._sha256(path) if path.exists() and path.is_file() else "",
                "note": note,
            }
        )
    return rows


def _scorefree_phase4p_best() -> dict[str, float]:
    path = PHASE4P / "mv_metric_scene_rows.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    out: dict[str, float] = {}
    for split, sub in df.groupby("dataset_split"):
        vals = pd.to_numeric(sub.get("ScoreFreeMatch50_scene"), errors="coerce")
        out[f"{split}_mean_ScoreFreeMatch50_scene"] = float(vals.mean()) if vals.notna().any() else 0.0
        out[f"{split}_max_ScoreFreeMatch50_scene"] = float(vals.max()) if vals.notna().any() else 0.0
    return out


def main() -> int:
    started = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    phase2c = _load_json(PHASE2C / "summary.json")
    phase4m = _load_json(PHASE4M / "summary.json")
    phase4n = _load_json(PHASE4N / "summary.json")
    phase4o = _load_json(PHASE4O / "summary.json")
    phase4p = _load_json(PHASE4P / "summary.json")
    phase5b = _load_json(PHASE5B / "summary.json")
    phase8b = _load_json(PHASE8B / "summary.json")
    baselines = p4h._phase0_baselines()
    f2_dev = baselines["F2_base_full_dev"]
    f2_holdout = baselines["F2_base_holdout"]
    dev_scene_gate = _num(f2_dev["MV_AP_scene"]) + 0.010
    dev_ap50_gate = _num(f2_dev["MV_AP50_scene"]) + 0.015
    hold_scene_gate = _num(f2_holdout["MV_AP_scene"]) + 0.006
    hold_ap50_gate = _num(f2_holdout["MV_AP50_scene"]) + 0.010

    local_pass = bool(phase2c.get("phase2c_pass")) and bool(phase2c.get("formal_claim_allowed"))
    scene_pass = bool(phase4p.get("phase4p_pass"))
    full_goal_achieved = bool(local_pass and scene_pass)
    decision = (
        "GO_LOCAL_AND_SCENE_PHASE2C_OVERLAP3"
        if full_goal_achieved
        else "GO_LOCAL_ONLY_PHASE2C_OVERLAP3__NO_GO_SCENE_STITCHING_AFTER_PHASE4M_TO_PHASE5B"
        if local_pass
        else "NO_GO_CHUNK_CAUSAL_LOCAL"
    )

    decision_rows = [
        {
            "schema_version": "stream4d_v100_phase8c_decision_row_v1",
            "phase_id": "v100_phase8c_continued_repair_decision_freeze",
            "decision": decision,
            "full_goal_achieved": full_goal_achieved,
            "local_claim_allowed": local_pass,
            "scene_claim_allowed": full_goal_achieved,
            "phase2c_overlap3_local_pass": phase2c.get("phase2c_pass"),
            "phase4m_temporal_maskview_pass": phase4m.get("phase4m_pass"),
            "phase4o_union_pass": phase4o.get("phase4o_pass"),
            "phase4p_multi_semantic_union_pass": phase4p.get("phase4p_pass"),
            "phase5b_d4rt_support_pass": phase5b.get("phase5b_pass"),
            "uses_gt_for_prediction": False,
        }
    ]
    scorefree = _scorefree_phase4p_best()
    metric_rows = [
        {
            "schema_version": "stream4d_v100_phase8c_metric_row_v1",
            "phase_id": "v100_phase8c_continued_repair_decision_freeze",
            "row_id": "phase2c_overlap3_local_claim",
            "dev_MV_AP_window": phase2c.get("dev_MV_AP_window"),
            "dev_MV_AP50_window": phase2c.get("dev_MV_AP50_window"),
            "holdout_MV_AP_window": phase2c.get("holdout_MV_AP_window"),
            "holdout_MV_AP50_window": phase2c.get("holdout_MV_AP50_window"),
            "overlap_transition_count": phase2c.get("overlap_transition_count"),
            "min_observed_overlap": phase2c.get("min_observed_overlap"),
            "max_observed_overlap": phase2c.get("max_observed_overlap"),
            "formal_claim_allowed": phase2c.get("formal_claim_allowed"),
        },
        {
            "schema_version": "stream4d_v100_phase8c_metric_row_v1",
            "phase_id": "v100_phase8c_continued_repair_decision_freeze",
            "row_id": "phase4m_temporal_maskview_no_go",
            "best_variant_id": phase4m.get("best_variant_id"),
            "dev_MV_AP_scene": phase4m.get("best_dev_MV_AP_scene"),
            "dev_MV_AP50_scene": phase4m.get("best_dev_MV_AP50_scene"),
            "holdout_MV_AP_scene": phase4m.get("best_holdout_MV_AP_scene"),
            "holdout_MV_AP50_scene": phase4m.get("best_holdout_MV_AP50_scene"),
            "formal_claim_allowed": False,
        },
        {
            "schema_version": "stream4d_v100_phase8c_metric_row_v1",
            "phase_id": "v100_phase8c_continued_repair_decision_freeze",
            "row_id": "phase4o_union_no_go",
            "best_variant_id": phase4o.get("best_variant_id"),
            "dev_MV_AP_scene": phase4o.get("best_dev_MV_AP_scene"),
            "dev_MV_AP50_scene": phase4o.get("best_dev_MV_AP50_scene"),
            "holdout_MV_AP_scene": phase4o.get("best_holdout_MV_AP_scene"),
            "holdout_MV_AP50_scene": phase4o.get("best_holdout_MV_AP50_scene"),
            "formal_claim_allowed": False,
        },
        {
            "schema_version": "stream4d_v100_phase8c_metric_row_v1",
            "phase_id": "v100_phase8c_continued_repair_decision_freeze",
            "row_id": "phase4p_best_current_scene_attempt",
            "best_variant_id": phase4p.get("best_variant_id"),
            "dev_MV_AP_scene": phase4p.get("best_dev_MV_AP_scene"),
            "dev_MV_AP50_scene": phase4p.get("best_dev_MV_AP50_scene"),
            "holdout_MV_AP_scene": phase4p.get("best_holdout_MV_AP_scene"),
            "holdout_MV_AP50_scene": phase4p.get("best_holdout_MV_AP50_scene"),
            "dev_scene_gate": dev_scene_gate,
            "dev_ap50_gate": dev_ap50_gate,
            "holdout_scene_gate": hold_scene_gate,
            "holdout_ap50_gate": hold_ap50_gate,
            "phase4p_pass": phase4p.get("phase4p_pass"),
            "formal_claim_allowed": False,
            **scorefree,
        },
        {
            "schema_version": "stream4d_v100_phase8c_metric_row_v1",
            "phase_id": "v100_phase8c_continued_repair_decision_freeze",
            "row_id": "phase5b_current_union_d4rt_support",
            "best_variant_id": phase5b.get("best_phase4p_variant_id"),
            "d4rt_real_minus_control_MV_AP_scene": phase5b.get("d4rt_real_minus_control_MV_AP_scene"),
            "holdout_d4rt_support_rate": next((row.get("d4rt_support_rate") for row in phase5b.get("summary_rows", []) if row.get("dataset_split") == "holdout"), ""),
            "holdout_d4rt_supported_edge_count": next((row.get("d4rt_supported_edge_count") for row in phase5b.get("summary_rows", []) if row.get("dataset_split") == "holdout"), ""),
            "holdout_union_edge_count": next((row.get("union_edge_count") for row in phase5b.get("summary_rows", []) if row.get("dataset_split") == "holdout"), ""),
            "phase5b_pass": phase5b.get("phase5b_pass"),
            "formal_claim_allowed": False,
        },
    ]
    repair_rows = [
        {
            "schema_version": "stream4d_v100_phase8c_repair_attempt_row_v1",
            "phase_id": "v100_phase8c_continued_repair_decision_freeze",
            "attempt_id": "phase4m_temporal_maskview",
            "decision": phase4m.get("decision"),
            "result": "No-Go; nearby-frame 2D mask IoU did not add useful holdout identity evidence.",
            "artifact": _rel(PHASE4M / "summary.json"),
        },
        {
            "schema_version": "stream4d_v100_phase8c_repair_attempt_row_v1",
            "phase_id": "v100_phase8c_continued_repair_decision_freeze",
            "attempt_id": "phase4n_gt_link_diagnostic",
            "decision": phase4n.get("decision"),
            "result": "Diagnostic-only; accepted links had moderate/high precision but low true-adjacent recall proxy.",
            "artifact": _rel(PHASE4N / "summary.json"),
            "uses_gt_for_prediction": False,
            "uses_gt_for_diagnostic": True,
        },
        {
            "schema_version": "stream4d_v100_phase8c_repair_attempt_row_v1",
            "phase_id": "v100_phase8c_continued_repair_decision_freeze",
            "attempt_id": "phase4o_union_history_evidence",
            "decision": phase4o.get("decision"),
            "result": "No-Go; semantic+Phase4h best union improved holdout but remained far below gates.",
            "artifact": _rel(PHASE4O / "summary.json"),
        },
        {
            "schema_version": "stream4d_v100_phase8c_repair_attempt_row_v1",
            "phase_id": "v100_phase8c_continued_repair_decision_freeze",
            "attempt_id": "phase4p_multi_semantic_union",
            "decision": phase4p.get("decision"),
            "result": "No-Go; all-HMS plus Phase4h best produced only marginal extra holdout gain.",
            "artifact": _rel(PHASE4P / "summary.json"),
        },
        {
            "schema_version": "stream4d_v100_phase8c_repair_attempt_row_v1",
            "phase_id": "v100_phase8c_continued_repair_decision_freeze",
            "attempt_id": "phase5b_current_union_d4rt_support",
            "decision": phase5b.get("decision"),
            "result": "No-Go; current best holdout union D4RT support rate was below verifier gate.",
            "artifact": _rel(PHASE5B / "summary.json"),
        },
    ]
    blocker_rows = [
        {
            "schema_version": "stream4d_v100_phase8c_blocker_row_v1",
            "phase_id": "v100_phase8c_continued_repair_decision_freeze",
            "blocker_id": "local_phase2c_achieved_scene_not_achieved",
            "status": "open_for_scene",
            "evidence": f"local_pass={local_pass}; phase4p_pass={phase4p.get('phase4p_pass')}",
            "conclusion": "Local F2 overlap3 claim is allowed; scene/local2history claim is not.",
        },
        {
            "schema_version": "stream4d_v100_phase8c_blocker_row_v1",
            "phase_id": "v100_phase8c_continued_repair_decision_freeze",
            "blocker_id": "best_current_scene_below_gates",
            "status": "open",
            "evidence": f"dev_scene={phase4p.get('best_dev_MV_AP_scene')} gate={dev_scene_gate}; holdout_scene={phase4p.get('best_holdout_MV_AP_scene')} gate={hold_scene_gate}; holdout_ap50={phase4p.get('best_holdout_MV_AP50_scene')} gate={hold_ap50_gate}",
            "conclusion": "Best current GT-free repair remains well below required scene gates.",
        },
        {
            "schema_version": "stream4d_v100_phase8c_blocker_row_v1",
            "phase_id": "v100_phase8c_continued_repair_decision_freeze",
            "blocker_id": "existing_identity_evidence_exhausted",
            "status": "open",
            "evidence": "Exact overlap, temporal mask-view, semantic memory, semantic+exact union, and multi-semantic union all failed.",
            "conclusion": "Existing F2 semantic/mask evidence recombination is insufficient; a genuinely new identity witness is required.",
        },
        {
            "schema_version": "stream4d_v100_phase8c_blocker_row_v1",
            "phase_id": "v100_phase8c_continued_repair_decision_freeze",
            "blocker_id": "d4rt_support_insufficient",
            "status": "open",
            "evidence": f"phase5b_decision={phase5b.get('decision')} holdout_support_rate={next((row.get('d4rt_support_rate') for row in phase5b.get('summary_rows', []) if row.get('dataset_split') == 'holdout'), '')}",
            "conclusion": "Current D4RT anchors cannot validate or expand enough of the best union links; keep D4RT diagnostic-only.",
        },
    ]

    decision_csv = OUT_DIR / "decision_rows.csv"
    metric_csv = OUT_DIR / "metric_rows.csv"
    repair_csv = OUT_DIR / "repair_attempt_rows.csv"
    blocker_csv = OUT_DIR / "blocker_rows.csv"
    artifact_csv = OUT_DIR / "artifact_manifest_rows.csv"
    summary_json = OUT_DIR / "summary.json"
    _write_csv(decision_csv, decision_rows)
    _write_csv(metric_csv, metric_rows)
    _write_csv(repair_csv, repair_rows)
    _write_csv(blocker_csv, blocker_rows)
    _write_csv(
        artifact_csv,
        _artifact_rows(
            [
                (decision_csv, "csv", "Phase8c decision rows"),
                (metric_csv, "csv", "Key metric evidence after continued repairs"),
                (repair_csv, "csv", "Continued repair attempt ledger"),
                (blocker_csv, "csv", "Remaining blockers"),
            ]
        ),
    )
    summary = {
        "schema_version": "stream4d_v100_phase8c_continued_repair_decision_freeze_summary_v1",
        "phase_id": "v100_phase8c_continued_repair_decision_freeze",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "runtime_sec": time.time() - started,
        "decision": decision,
        "previous_decision": phase8b.get("decision"),
        "full_goal_achieved": full_goal_achieved,
        "local_claim_allowed": local_pass,
        "scene_claim_allowed": full_goal_achieved,
        "best_current_scene_attempt": {
            "source_id": "phase4p_multi_semantic_union",
            "variant_id": phase4p.get("best_variant_id"),
            "dev_MV_AP_scene": phase4p.get("best_dev_MV_AP_scene"),
            "dev_MV_AP50_scene": phase4p.get("best_dev_MV_AP50_scene"),
            "holdout_MV_AP_scene": phase4p.get("best_holdout_MV_AP_scene"),
            "holdout_MV_AP50_scene": phase4p.get("best_holdout_MV_AP50_scene"),
            "dev_scene_gate": dev_scene_gate,
            "dev_ap50_gate": dev_ap50_gate,
            "holdout_scene_gate": hold_scene_gate,
            "holdout_ap50_gate": hold_ap50_gate,
        },
        "phase2c_overlap3_local": {
            "dev_MV_AP_window": phase2c.get("dev_MV_AP_window"),
            "dev_MV_AP50_window": phase2c.get("dev_MV_AP50_window"),
            "holdout_MV_AP_window": phase2c.get("holdout_MV_AP_window"),
            "holdout_MV_AP50_window": phase2c.get("holdout_MV_AP50_window"),
            "formal_claim_allowed": phase2c.get("formal_claim_allowed"),
        },
        "phase5b_d4rt_support": {
            "decision": phase5b.get("decision"),
            "summary_rows": phase5b.get("summary_rows"),
        },
        "uses_gt_for_prediction": False,
        "outputs": {
            "summary": _rel(summary_json),
            "decision_rows": _rel(decision_csv),
            "metric_rows": _rel(metric_csv),
            "repair_attempt_rows": _rel(repair_csv),
            "blocker_rows": _rel(blocker_csv),
            "artifact_manifest_rows": _rel(artifact_csv),
        },
    }
    _write_json(summary_json, summary)
    print(json.dumps(p4h._jsonable(summary), indent=2, sort_keys=True))
    return 0 if full_goal_achieved else 2


if __name__ == "__main__":
    raise SystemExit(main())
