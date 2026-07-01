#!/usr/bin/env python3
"""Freeze v100 decision after Phase4q/4r and Phase5c repairs."""

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
OUT_DIR = AUDIT_ROOT / "v100_phase8d_continued_repair_decision_freeze"
PHASE2C = AUDIT_ROOT / "v100_phase2c_overlap3_local_repair"
PHASE4P = AUDIT_ROOT / "v100_phase4p_multi_semantic_union_repair"
PHASE4Q = AUDIT_ROOT / "v100_phase4q_phase2c_position_history_repair"
PHASE4R = AUDIT_ROOT / "v100_phase4r_position_union_repair"
PHASE5B = AUDIT_ROOT / "v100_phase5b_current_union_d4rt_support_audit"
PHASE5C = AUDIT_ROOT / "v100_phase5c_da3_broad_split_repair"
PHASE8C = AUDIT_ROOT / "v100_phase8c_continued_repair_decision_freeze"
PHASE5C_SCOPE_BUG = AUDIT_ROOT / "v100_phase5c_da3_broad_split_repair_source_scope_bug_20260701_115050"


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
                "schema_version": "stream4d_v100_phase8d_artifact_manifest_row_v1",
                "phase_id": "v100_phase8d_continued_repair_decision_freeze",
                "artifact_path": _rel(path),
                "artifact_type": kind,
                "size_bytes": path.stat().st_size if path.exists() else 0,
                "sha256": p4h._sha256(path) if path.exists() and path.is_file() else "",
                "note": note,
            }
        )
    return rows


def _scene_attempts(phase4p: dict[str, Any], phase4r: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "source_id": "phase4p_multi_semantic_union",
            "variant_id": phase4p.get("best_variant_id"),
            "dev_MV_AP_scene": _num(phase4p.get("best_dev_MV_AP_scene")),
            "dev_MV_AP50_scene": _num(phase4p.get("best_dev_MV_AP50_scene")),
            "holdout_MV_AP_scene": _num(phase4p.get("best_holdout_MV_AP_scene")),
            "holdout_MV_AP50_scene": _num(phase4p.get("best_holdout_MV_AP50_scene")),
            "artifact": _rel(PHASE4P / "summary.json"),
        },
        {
            "source_id": "phase4r_position_union",
            "variant_id": phase4r.get("best_variant_id"),
            "dev_MV_AP_scene": _num(phase4r.get("best_dev_MV_AP_scene")),
            "dev_MV_AP50_scene": _num(phase4r.get("best_dev_MV_AP50_scene")),
            "holdout_MV_AP_scene": _num(phase4r.get("best_holdout_MV_AP_scene")),
            "holdout_MV_AP50_scene": _num(phase4r.get("best_holdout_MV_AP50_scene")),
            "artifact": _rel(PHASE4R / "summary.json"),
        },
    ]


def main() -> int:
    started = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    phase2c = _load_json(PHASE2C / "summary.json")
    phase4p = _load_json(PHASE4P / "summary.json")
    phase4q = _load_json(PHASE4Q / "summary.json")
    phase4r = _load_json(PHASE4R / "summary.json")
    phase5b = _load_json(PHASE5B / "summary.json")
    phase5c = _load_json(PHASE5C / "summary.json")
    phase8c = _load_json(PHASE8C / "summary.json")
    baselines = p4h._phase0_baselines()
    f2_dev = baselines["F2_base_full_dev"]
    f2_holdout = baselines["F2_base_holdout"]
    dev_scene_gate = _num(f2_dev["MV_AP_scene"]) + 0.010
    dev_ap50_gate = _num(f2_dev["MV_AP50_scene"]) + 0.015
    hold_scene_gate = _num(f2_holdout["MV_AP_scene"]) + 0.006
    hold_ap50_gate = _num(f2_holdout["MV_AP50_scene"]) + 0.010

    local_pass = bool(phase2c.get("phase2c_pass")) and bool(phase2c.get("formal_claim_allowed"))
    attempts = _scene_attempts(phase4p, phase4r)
    best_scene = max(
        attempts,
        key=lambda row: (
            _num(row.get("holdout_MV_AP_scene")),
            _num(row.get("holdout_MV_AP50_scene")),
            _num(row.get("dev_MV_AP_scene")),
            _num(row.get("dev_MV_AP50_scene")),
        ),
    )
    scene_pass = bool(
        _num(best_scene["dev_MV_AP_scene"]) >= dev_scene_gate
        and _num(best_scene["dev_MV_AP50_scene"]) >= dev_ap50_gate
        and _num(best_scene["holdout_MV_AP_scene"]) >= hold_scene_gate
        and _num(best_scene["holdout_MV_AP50_scene"]) >= hold_ap50_gate
    )
    full_goal_achieved = bool(local_pass and scene_pass)
    decision = (
        "GO_LOCAL_AND_SCENE_PHASE2C_OVERLAP3_AFTER_PHASE4R_OR_PHASE5C"
        if full_goal_achieved
        else "GO_LOCAL_ONLY_PHASE2C_OVERLAP3__NO_GO_SCENE_STITCHING_AFTER_PHASE4Q_R_AND_PHASE5C"
        if local_pass
        else "NO_GO_CHUNK_CAUSAL_LOCAL"
    )

    decision_rows = [
        {
            "schema_version": "stream4d_v100_phase8d_decision_row_v1",
            "phase_id": "v100_phase8d_continued_repair_decision_freeze",
            "decision": decision,
            "previous_decision": phase8c.get("decision"),
            "full_goal_achieved": full_goal_achieved,
            "local_claim_allowed": local_pass,
            "scene_claim_allowed": full_goal_achieved,
            "phase2c_overlap3_local_pass": phase2c.get("phase2c_pass"),
            "phase4r_position_union_pass": phase4r.get("phase4r_pass"),
            "phase5c_da3_broad_split_pass": phase5c.get("phase5c_pass"),
            "uses_gt_for_prediction": False,
        }
    ]
    metric_rows = [
        {
            "schema_version": "stream4d_v100_phase8d_metric_row_v1",
            "phase_id": "v100_phase8d_continued_repair_decision_freeze",
            "row_id": "phase2c_overlap3_local_claim",
            "dev_MV_AP_window": phase2c.get("dev_MV_AP_window"),
            "dev_MV_AP50_window": phase2c.get("dev_MV_AP50_window"),
            "holdout_MV_AP_window": phase2c.get("holdout_MV_AP_window"),
            "holdout_MV_AP50_window": phase2c.get("holdout_MV_AP50_window"),
            "formal_claim_allowed": phase2c.get("formal_claim_allowed"),
        },
        {
            "schema_version": "stream4d_v100_phase8d_metric_row_v1",
            "phase_id": "v100_phase8d_continued_repair_decision_freeze",
            "row_id": "phase4q_position_history_no_go",
            "best_variant_id": phase4q.get("best_variant_id"),
            "dev_MV_AP_scene": phase4q.get("best_dev_MV_AP_scene"),
            "dev_MV_AP50_scene": phase4q.get("best_dev_MV_AP50_scene"),
            "holdout_MV_AP_scene": phase4q.get("best_holdout_MV_AP_scene"),
            "holdout_MV_AP50_scene": phase4q.get("best_holdout_MV_AP50_scene"),
            "phase4q_pass": phase4q.get("phase4_pass", phase4q.get("phase4q_pass")),
            "formal_claim_allowed": False,
        },
        {
            "schema_version": "stream4d_v100_phase8d_metric_row_v1",
            "phase_id": "v100_phase8d_continued_repair_decision_freeze",
            "row_id": "phase4r_best_current_scene_attempt",
            "best_variant_id": phase4r.get("best_variant_id"),
            "dev_MV_AP_scene": phase4r.get("best_dev_MV_AP_scene"),
            "dev_MV_AP50_scene": phase4r.get("best_dev_MV_AP50_scene"),
            "holdout_MV_AP_scene": phase4r.get("best_holdout_MV_AP_scene"),
            "holdout_MV_AP50_scene": phase4r.get("best_holdout_MV_AP50_scene"),
            "dev_scene_gate": dev_scene_gate,
            "dev_ap50_gate": dev_ap50_gate,
            "holdout_scene_gate": hold_scene_gate,
            "holdout_ap50_gate": hold_ap50_gate,
            "phase4r_pass": phase4r.get("phase4r_pass"),
            "formal_claim_allowed": False,
        },
        {
            "schema_version": "stream4d_v100_phase8d_metric_row_v1",
            "phase_id": "v100_phase8d_continued_repair_decision_freeze",
            "row_id": "phase5c_da3_broad_split_no_go",
            "best_variant_id": phase5c.get("best_variant_id"),
            "best_split_variant_id": phase5c.get("best_split_variant_id"),
            "best_holdout_MV_AP_scene": phase5c.get("best_holdout_MV_AP_scene"),
            "best_split_holdout_MV_AP_scene": phase5c.get("best_split_holdout_MV_AP_scene"),
            "split_component_row_count": phase5c.get("split_component_row_count"),
            "broad_mask_count": json.dumps(phase5c.get("broad_mask_count", {}), sort_keys=True),
            "phase5c_pass": phase5c.get("phase5c_pass"),
            "formal_claim_allowed": False,
        },
    ]
    repair_rows = [
        {
            "schema_version": "stream4d_v100_phase8d_repair_attempt_row_v1",
            "phase_id": "v100_phase8d_continued_repair_decision_freeze",
            "attempt_id": "phase4q_scene_position_history",
            "decision": phase4q.get("decision"),
            "result": "No-Go; position continuity alone was weak and damaged local scope when used directly.",
            "artifact": _rel(PHASE4Q / "summary.json"),
        },
        {
            "schema_version": "stream4d_v100_phase8d_repair_attempt_row_v1",
            "phase_id": "v100_phase8d_continued_repair_decision_freeze",
            "attempt_id": "phase4r_phase4p_plus_position_union",
            "decision": phase4r.get("decision"),
            "result": "No-Go; HMP4 added a small holdout gain but remained far below scene gates.",
            "artifact": _rel(PHASE4R / "summary.json"),
        },
        {
            "schema_version": "stream4d_v100_phase8d_repair_attempt_row_v1",
            "phase_id": "v100_phase8d_continued_repair_decision_freeze",
            "attempt_id": "phase5c_da3_broad_split",
            "decision": phase5c.get("decision"),
            "result": "No-Go after scope fix; DA3 median-depth split did not beat no-split baseline.",
            "artifact": _rel(PHASE5C / "summary.json"),
            "bug_backup_artifact": _rel(PHASE5C_SCOPE_BUG),
        },
        {
            "schema_version": "stream4d_v100_phase8d_repair_attempt_row_v1",
            "phase_id": "v100_phase8d_continued_repair_decision_freeze",
            "attempt_id": "phase5b_d4rt_support_context",
            "decision": phase5b.get("decision"),
            "result": "No-Go; D4RT support rate was insufficient for the previous best union.",
            "artifact": _rel(PHASE5B / "summary.json"),
        },
    ]
    blocker_rows = [
        {
            "schema_version": "stream4d_v100_phase8d_blocker_row_v1",
            "phase_id": "v100_phase8d_continued_repair_decision_freeze",
            "blocker_id": "full_goal_not_achieved",
            "status": "open",
            "evidence": f"local_pass={local_pass}; scene_pass={scene_pass}; best_scene_source={best_scene['source_id']}",
            "conclusion": "Only Phase2c local overlap3 claim is allowed; full v100 scene/local2history target remains No-Go.",
        },
        {
            "schema_version": "stream4d_v100_phase8d_blocker_row_v1",
            "phase_id": "v100_phase8d_continued_repair_decision_freeze",
            "blocker_id": "best_scene_attempt_below_holdout_gates",
            "status": "open",
            "evidence": f"holdout_scene={best_scene['holdout_MV_AP_scene']} gate={hold_scene_gate}; holdout_ap50={best_scene['holdout_MV_AP50_scene']} gate={hold_ap50_gate}",
            "conclusion": "Phase4r is the best current attempt but still lacks enough holdout scene AP and AP50.",
        },
        {
            "schema_version": "stream4d_v100_phase8d_blocker_row_v1",
            "phase_id": "v100_phase8d_continued_repair_decision_freeze",
            "blocker_id": "da3_median_depth_split_quality",
            "status": "closed_as_no_go_for_this_method",
            "evidence": f"split_component_row_count={phase5c.get('split_component_row_count')}; best_variant={phase5c.get('best_variant_id')}; best_split={phase5c.get('best_split_variant_id')}",
            "conclusion": "Provider availability was not the blocker for median-depth split; split quality and identity usefulness were insufficient.",
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
                (decision_csv, "csv", "Phase8d decision rows"),
                (metric_csv, "csv", "Key metric evidence after Phase4q/4r and Phase5c"),
                (repair_csv, "csv", "Continued repair attempt ledger"),
                (blocker_csv, "csv", "Remaining blockers"),
            ]
        ),
    )
    summary = {
        "schema_version": "stream4d_v100_phase8d_continued_repair_decision_freeze_summary_v1",
        "phase_id": "v100_phase8d_continued_repair_decision_freeze",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "runtime_sec": time.time() - started,
        "decision": decision,
        "previous_decision": phase8c.get("decision"),
        "full_goal_achieved": full_goal_achieved,
        "local_claim_allowed": local_pass,
        "scene_claim_allowed": full_goal_achieved,
        "best_current_scene_attempt": {
            **best_scene,
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
        "phase5c_da3_broad_split": {
            "decision": phase5c.get("decision"),
            "best_variant_id": phase5c.get("best_variant_id"),
            "best_split_variant_id": phase5c.get("best_split_variant_id"),
            "split_component_row_count": phase5c.get("split_component_row_count"),
            "broad_mask_count": phase5c.get("broad_mask_count"),
            "bug_backup_artifact": _rel(PHASE5C_SCOPE_BUG),
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
