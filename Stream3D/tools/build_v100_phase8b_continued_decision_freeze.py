#!/usr/bin/env python3
"""Freeze v100 continued decision after Phase2c/Phase4h-k repairs."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


STREAM3D_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = STREAM3D_ROOT.parent
AUDIT_ROOT = STREAM3D_ROOT / "outputs/audit"
OUT_DIR = AUDIT_ROOT / "v100_phase8b_continued_decision_freeze"

PHASE2B = AUDIT_ROOT / "v100_phase2b_overlap_contract_audit"
PHASE2C = AUDIT_ROOT / "v100_phase2c_overlap3_local_repair"
PHASE4H = AUDIT_ROOT / "v100_phase4h_overlap3_exact_history_memory"
PHASE4K = AUDIT_ROOT / "v100_phase4k_phase2c_semantic_scene_repair"
PHASE4L = AUDIT_ROOT / "v100_phase4l_phase2c_scope_fixed_scene_decision"
PHASE4G = AUDIT_ROOT / "v100_phase4g_scene_score_calibration"
PHASE5 = AUDIT_ROOT / "v100_phase5_da3_d4rt_verifier_audit"
V99_ORACLE = AUDIT_ROOT / "v99_phase10q_gt_oracle_scene_stitch_diagnostic"


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


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _num(value: Any, default: float = 0.0) -> float:
    try:
        if value in ("", None):
            return float(default)
        out = float(value)
    except Exception:
        return float(default)
    return out if math.isfinite(out) else float(default)


def _scorefree_max_rows() -> list[dict[str, Any]]:
    rows: list[pd.DataFrame] = []
    for source_id, path in [
        ("phase4h_overlap3_exact", PHASE4H / "variant_metric_rows.csv"),
        ("phase4k_phase2c_semantic", PHASE4K / "variant_metric_rows.csv"),
        ("phase4l_scope_fixed", PHASE4L / "variant_metric_rows.csv"),
    ]:
        if not path.exists():
            continue
        df = pd.read_csv(path)
        df["source_id"] = source_id
        rows.append(df)
    if not rows:
        return []
    df = pd.concat(rows, ignore_index=True)
    out: list[dict[str, Any]] = []
    for split, sub in df.groupby("dataset_split"):
        top_scene = sub.sort_values(["MV_AP_scene", "MV_AP50_scene"], ascending=False).head(1).iloc[0]
        top_sf = sub.sort_values(["ScoreFreeMatch50_scene"], ascending=False).head(1).iloc[0]
        out.append(
            {
                "schema_version": "stream4d_v100_phase8b_scorefree_row_v1",
                "phase_id": "v100_phase8b_continued_decision_freeze",
                "dataset_split": split,
                "best_scene_source_id": top_scene.get("source_id"),
                "best_scene_variant_id": top_scene.get("variant_id"),
                "best_MV_AP_scene": _num(top_scene.get("MV_AP_scene")),
                "best_MV_AP50_scene": _num(top_scene.get("MV_AP50_scene")),
                "best_scorefree_source_id": top_sf.get("source_id"),
                "best_scorefree_variant_id": top_sf.get("variant_id"),
                "best_ScoreFreeMatch50_scene": _num(top_sf.get("ScoreFreeMatch50_scene")),
            }
        )
    return out


def _artifact_rows(paths: list[tuple[Path, str, str]]) -> list[dict[str, Any]]:
    return [
        {
            "schema_version": "stream4d_v100_phase8b_artifact_manifest_row_v1",
            "phase_id": "v100_phase8b_continued_decision_freeze",
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

    phase2b = _load_json(PHASE2B / "summary.json")
    phase2c = _load_json(PHASE2C / "summary.json")
    phase4h = _load_json(PHASE4H / "summary.json")
    phase4k = _load_json(PHASE4K / "summary.json")
    phase4l = _load_json(PHASE4L / "summary.json")
    phase5 = _load_json(PHASE5 / "summary.json")
    oracle = _load_json(V99_ORACLE / "summary.json")
    phase4g = _load_json(PHASE4G / "summary.json") if (PHASE4G / "summary.json").exists() else {}
    scorefree_rows = _scorefree_max_rows()

    full_scene_pass = bool(phase4l.get("phase4c_pass"))
    local_pass = bool(phase2c.get("phase2c_pass")) and bool(phase2c.get("formal_claim_allowed"))
    decision = (
        "GO_LOCAL_AND_SCENE_PHASE2C_OVERLAP3"
        if local_pass and full_scene_pass
        else "GO_LOCAL_ONLY_PHASE2C_OVERLAP3__NO_GO_SCENE_STITCHING"
        if local_pass
        else "NO_GO_CHUNK_CAUSAL_LOCAL"
    )
    decision_rows = [
        {
            "schema_version": "stream4d_v100_phase8b_decision_row_v1",
            "phase_id": "v100_phase8b_continued_decision_freeze",
            "decision": decision,
            "full_goal_achieved": bool(local_pass and full_scene_pass),
            "local_claim_allowed": local_pass,
            "scene_claim_allowed": bool(local_pass and full_scene_pass),
            "phase2b_old_overlap_contract_pass": phase2b.get("phase2b_pass"),
            "phase2c_overlap3_local_pass": phase2c.get("phase2c_pass"),
            "phase4h_overlap_exact_pass": phase4h.get("phase4h_pass"),
            "phase4k_semantic_pass": phase4k.get("phase4_pass"),
            "phase4l_scope_fixed_pass": phase4l.get("phase4c_pass"),
            "phase5_da3_d4rt_pass": phase5.get("phase5_pass"),
            "uses_gt_for_prediction": False,
        }
    ]
    metric_rows = [
        {
            "schema_version": "stream4d_v100_phase8b_metric_row_v1",
            "phase_id": "v100_phase8b_continued_decision_freeze",
            "row_id": "phase2c_overlap3_local",
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
            "schema_version": "stream4d_v100_phase8b_metric_row_v1",
            "phase_id": "v100_phase8b_continued_decision_freeze",
            "row_id": "phase4l_best_phase2c_compatible_scene",
            "best_source_id": phase4l.get("best_source_id"),
            "best_variant_id": phase4l.get("best_variant_id"),
            "dev_MV_AP_scene": phase4l.get("best_dev_MV_AP_scene"),
            "dev_MV_AP50_scene": phase4l.get("best_dev_MV_AP50_scene"),
            "holdout_MV_AP_scene": phase4l.get("best_holdout_MV_AP_scene"),
            "holdout_MV_AP50_scene": phase4l.get("best_holdout_MV_AP50_scene"),
            "adapter_scope_local_drop": phase4l.get("adapter_scope_local_drop"),
            "formal_claim_allowed": False,
        },
        {
            "schema_version": "stream4d_v100_phase8b_metric_row_v1",
            "phase_id": "v100_phase8b_continued_decision_freeze",
            "row_id": "phase4h_best_dev_but_holdout_fail",
            "best_variant_id": phase4h.get("best_variant_id"),
            "dev_MV_AP_scene": phase4h.get("best_dev_MV_AP_scene"),
            "dev_MV_AP50_scene": phase4h.get("best_dev_MV_AP50_scene"),
            "holdout_MV_AP_scene": phase4h.get("best_holdout_MV_AP_scene"),
            "holdout_MV_AP50_scene": phase4h.get("best_holdout_MV_AP50_scene"),
            "formal_claim_allowed": False,
        },
        {
            "schema_version": "stream4d_v100_phase8b_metric_row_v1",
            "phase_id": "v100_phase8b_continued_decision_freeze",
            "row_id": "phase4g_old_non_phase2c_score_calibration_reference",
            "best_source_id": phase4g.get("best_source_id", ""),
            "best_variant_id": phase4g.get("best_variant_id", ""),
            "dev_MV_AP_scene": phase4g.get("best_dev_MV_AP_scene", ""),
            "dev_MV_AP50_scene": phase4g.get("best_dev_MV_AP50_scene", ""),
            "holdout_MV_AP_scene": phase4g.get("best_holdout_MV_AP_scene", ""),
            "holdout_MV_AP50_scene": phase4g.get("best_holdout_MV_AP50_scene", ""),
            "formal_claim_allowed": False,
            "note": "Reference only; source rows are not the strict Phase2c overlap3 artifact.",
        },
        {
            "schema_version": "stream4d_v100_phase8b_metric_row_v1",
            "phase_id": "v100_phase8b_continued_decision_freeze",
            "row_id": "v99_gt_oracle_diagnostic_upper_bound",
            "oracle_MV_AP_scene": oracle.get("oracle_MV_AP_scene"),
            "oracle_MV_AP50_scene": oracle.get("oracle_MV_AP50_scene"),
            "uses_gt_for_prediction": oracle.get("uses_gt_for_prediction"),
            "formal_claim_allowed": False,
            "note": "Diagnostic-only GT identity upper bound; not a method result.",
        },
    ]
    blocker_rows = [
        {
            "schema_version": "stream4d_v100_phase8b_blocker_row_v1",
            "phase_id": "v100_phase8b_continued_decision_freeze",
            "blocker_id": "old_phase2_overlap3_contract_false",
            "status": "repaired_by_phase2c",
            "evidence": f"Phase2b decision={phase2b.get('decision')} min/max overlap={phase2b.get('min_observed_overlap')}/{phase2b.get('max_observed_overlap')}; Phase2c min/max overlap={phase2c.get('min_observed_overlap')}/{phase2c.get('max_observed_overlap')}",
            "conclusion": "Old Phase2 import cannot support overlap3 formal claim; Phase2c is the repaired local artifact.",
        },
        {
            "schema_version": "stream4d_v100_phase8b_blocker_row_v1",
            "phase_id": "v100_phase8b_continued_decision_freeze",
            "blocker_id": "phase2c_scene_identity_no_go",
            "status": "open",
            "evidence": f"Phase4l best holdout scene={phase4l.get('best_holdout_MV_AP_scene')} AP50={phase4l.get('best_holdout_MV_AP50_scene')} with local drop={phase4l.get('adapter_scope_local_drop')}",
            "conclusion": "Local-window AP is protected after adapter-scope separation, but scene identity quality remains below gates.",
        },
        {
            "schema_version": "stream4d_v100_phase8b_blocker_row_v1",
            "phase_id": "v100_phase8b_continued_decision_freeze",
            "blocker_id": "score_calibration_insufficient_for_phase2c_holdout",
            "status": "open",
            "evidence": "; ".join(
                f"{row['dataset_split']}: best_scorefree50={row['best_ScoreFreeMatch50_scene']} ({row['best_scorefree_variant_id']})"
                for row in scorefree_rows
            ),
            "conclusion": "Phase2c-compatible holdout score-free AP50 evidence is far below the AP50 scene gate, so score calibration alone is not a credible fix.",
        },
        {
            "schema_version": "stream4d_v100_phase8b_blocker_row_v1",
            "phase_id": "v100_phase8b_continued_decision_freeze",
            "blocker_id": "geometry_verifier_not_promoted",
            "status": "open",
            "evidence": f"Phase5 decision={phase5.get('decision')} phase5_pass={phase5.get('phase5_pass')}",
            "conclusion": "DA3/D4RT verifier evidence remains diagnostic-only and does not clear scene gates.",
        },
    ]

    decision_csv = OUT_DIR / "decision_rows.csv"
    metric_csv = OUT_DIR / "metric_rows.csv"
    blocker_csv = OUT_DIR / "blocker_rows.csv"
    scorefree_csv = OUT_DIR / "scorefree_rows.csv"
    artifact_csv = OUT_DIR / "artifact_manifest_rows.csv"
    summary_json = OUT_DIR / "summary.json"
    _write_csv(decision_csv, decision_rows)
    _write_csv(metric_csv, metric_rows)
    _write_csv(blocker_csv, blocker_rows)
    _write_csv(scorefree_csv, scorefree_rows)
    _write_csv(
        artifact_csv,
        _artifact_rows(
            [
                (decision_csv, "csv", "continued decision rows"),
                (metric_csv, "csv", "key metric evidence"),
                (blocker_csv, "csv", "open and repaired blockers"),
                (scorefree_csv, "csv", "score-free AP50 evidence for score calibration triage"),
            ]
        ),
    )
    summary = {
        "schema_version": "stream4d_v100_phase8b_continued_decision_freeze_summary_v1",
        "phase_id": "v100_phase8b_continued_decision_freeze",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "runtime_sec": time.time() - started,
        "decision": decision,
        "full_goal_achieved": bool(local_pass and full_scene_pass),
        "local_claim_allowed": local_pass,
        "scene_claim_allowed": bool(local_pass and full_scene_pass),
        "phase2c_overlap3_local_pass": phase2c.get("phase2c_pass"),
        "phase4l_phase2c_scene_pass": phase4l.get("phase4c_pass"),
        "best_phase2c_compatible_scene": {
            "source_id": phase4l.get("best_source_id"),
            "variant_id": phase4l.get("best_variant_id"),
            "dev_MV_AP_scene": phase4l.get("best_dev_MV_AP_scene"),
            "dev_MV_AP50_scene": phase4l.get("best_dev_MV_AP50_scene"),
            "holdout_MV_AP_scene": phase4l.get("best_holdout_MV_AP_scene"),
            "holdout_MV_AP50_scene": phase4l.get("best_holdout_MV_AP50_scene"),
        },
        "outputs": {
            "summary": _rel(summary_json),
            "decision_rows": _rel(decision_csv),
            "metric_rows": _rel(metric_csv),
            "blocker_rows": _rel(blocker_csv),
            "scorefree_rows": _rel(scorefree_csv),
            "artifact_manifest_rows": _rel(artifact_csv),
        },
    }
    _write_json(summary_json, summary)
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))
    return 0 if local_pass and full_scene_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
