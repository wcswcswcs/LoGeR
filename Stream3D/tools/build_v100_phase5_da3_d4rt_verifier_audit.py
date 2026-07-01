#!/usr/bin/env python3
"""Audit DA3/D4RT verifier viability for v100 Phase5."""

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


STREAM3D_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = STREAM3D_ROOT.parent
AUDIT_ROOT = STREAM3D_ROOT / "outputs/audit"
OUT_DIR = AUDIT_ROOT / "v100_phase5_da3_d4rt_verifier_audit"

PHASE4_MAIN = AUDIT_ROOT / "v100_phase4_history_memory"
PHASE4_REPAIR = AUDIT_ROOT / "v100_phase4b_history_memory_repair"
V99_DA3 = AUDIT_ROOT / "v99_phase4_f2_da3_link_verifier"
V99_D4RT_AH = AUDIT_ROOT / "v99_phase10ah_prefix_sim3_aligned_anchor_scene_stitch"
V99_D4RT_AI = AUDIT_ROOT / "v99_phase10ai_prefix_sim3_d4rt_semantic_scene_repair_cupy"
V99_SIM3 = AUDIT_ROOT / "v99_phase10ag_prefix_da3_d4rt_sim3_alignment"


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


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _canon_oid(oid: Any) -> str:
    text = str(oid)
    for prefix in ("dev:", "holdout:"):
        if text.startswith(prefix):
            return text[len(prefix) :]
    return text


def _d4rt_candidate_lookup() -> dict[frozenset[str], dict[str, Any]]:
    lookup: dict[frozenset[str], dict[str, Any]] = {}
    for row in _read_csv(V99_D4RT_AH / "local2history_candidate_rows.csv"):
        family = str(row.get("anchor_family", ""))
        if family not in {"real_R20", "real_R40"}:
            continue
        a = _canon_oid(row.get("mv_object_id_a", ""))
        b = _canon_oid(row.get("mv_object_id_b", ""))
        key = frozenset({a, b})
        overlap = _num(row.get("object_anchor_overlap"))
        old = lookup.get(key)
        if old is None or overlap > _num(old.get("object_anchor_overlap")):
            lookup[key] = dict(row)
    return lookup


def _phase4_link_support_rows(phase4_dir: Path, variant_id: str) -> list[dict[str, Any]]:
    link_df = pd.read_parquet(phase4_dir / "chunk_object_history_link_rows.parquet")
    hist_df = pd.read_parquet(phase4_dir / "history_object_rows.parquet")
    d4rt_lookup = _d4rt_candidate_lookup()
    local_by_history = {
        str(row.history_id): [_canon_oid(v) for v in str(row.local_object_ids).split(";") if v]
        for row in hist_df.itertuples(index=False)
        if str(row.variant_id) == variant_id and str(row.dataset_split) == "holdout"
    }
    rows: list[dict[str, Any]] = []
    accepted = link_df[
        (link_df["variant_id"] == variant_id)
        & (link_df["dataset_split"] == "holdout")
        & (link_df["action"] == "accept_link")
    ]
    for row in accepted.itertuples(index=False):
        current = _canon_oid(row.chunk_object_id)
        hist_locals = [oid for oid in local_by_history.get(str(row.history_id), []) if oid != current]
        best: dict[str, Any] | None = None
        best_partner = ""
        for prev in hist_locals:
            cand = d4rt_lookup.get(frozenset({current, prev}))
            if cand is None:
                continue
            if best is None or _num(cand.get("object_anchor_overlap")) > _num(best.get("object_anchor_overlap")):
                best = cand
                best_partner = prev
        overlap = _num(best.get("object_anchor_overlap")) if best else 0.0
        rows.append(
            {
                "schema_version": "stream4d_v100_phase5_phase4_link_verifier_row_v1",
                "phase_id": "v100_phase5_da3_d4rt_verifier_audit",
                "source_phase4_dir": _rel(phase4_dir),
                "variant_id": variant_id,
                "dataset_split": "holdout",
                "chunk_object_id": str(row.chunk_object_id),
                "history_id": str(row.history_id),
                "semantic_link_score": row.link_score,
                "semantic_link_margin": row.link_margin,
                "d4rt_supported": bool(best is not None and overlap >= 0.20),
                "best_d4rt_overlap": overlap,
                "best_d4rt_partner_local_object_id": best_partner,
                "best_d4rt_anchor_family": best.get("anchor_family", "") if best else "",
                "best_d4rt_shared_anchor_count": best.get("shared_anchor_count", "") if best else "",
                "uses_gt_for_prediction": False,
                "uses_future": False,
            }
        )
    return rows


def _variant_rows() -> list[dict[str, Any]]:
    da3 = _load_json(V99_DA3 / "summary.json")
    d4rt_ah = _load_json(V99_D4RT_AH / "summary.json")
    d4rt_ai = _load_json(V99_D4RT_AI / "summary.json")
    sim3 = _load_json(V99_SIM3 / "summary.json")
    phase4 = _load_json(PHASE4_MAIN / "summary.json")
    phase4b = _load_json(PHASE4_REPAIR / "summary.json")
    rows = [
        {
            "schema_version": "stream4d_v100_phase5_variant_row_v1",
            "phase_id": "v100_phase5_da3_d4rt_verifier_audit",
            "variant_id": "G0_F2_history_memory",
            "status": "no_go_input",
            "source_summary": _rel(PHASE4_MAIN / "summary.json"),
            "MV_AP_scene_holdout": phase4.get("best_holdout_MV_AP_scene"),
            "MV_AP50_scene_holdout": phase4.get("best_holdout_MV_AP50_scene"),
            "MV_AP_window_holdout": phase4.get("best_holdout_MV_AP_window"),
            "phase4_pass": phase4.get("phase4_pass"),
            "diagnostic_only": True,
        },
        {
            "schema_version": "stream4d_v100_phase5_variant_row_v1",
            "phase_id": "v100_phase5_da3_d4rt_verifier_audit",
            "variant_id": "G0b_F2_history_memory_repair_local",
            "status": "no_go_input",
            "source_summary": _rel(PHASE4_REPAIR / "summary.json"),
            "MV_AP_scene_holdout": phase4b.get("best_holdout_MV_AP_scene"),
            "MV_AP50_scene_holdout": phase4b.get("best_holdout_MV_AP50_scene"),
            "MV_AP_window_holdout": phase4b.get("best_holdout_MV_AP_window"),
            "phase4_pass": phase4b.get("phase4_pass"),
            "diagnostic_only": True,
        },
        {
            "schema_version": "stream4d_v100_phase5_variant_row_v1",
            "phase_id": "v100_phase5_da3_d4rt_verifier_audit",
            "variant_id": "G1_F2_plus_DA3_link_verifier",
            "status": "diagnostic_only_not_promoted",
            "source_summary": _rel(V99_DA3 / "summary.json"),
            "provider_available": True,
            "v99_decision": da3.get("decision"),
            "delta_scene_vs_phase2": da3.get("delta_best_real_vs_phase2_scene"),
            "delta_window_vs_phase2": da3.get("delta_best_real_vs_phase2_window"),
            "best_real_MV_AP_scene": da3.get("best_real_MV_AP_scene"),
            "best_real_MV_AP_window": da3.get("best_real_MV_AP_window"),
            "uses_gt_for_prediction": da3.get("uses_gt_for_prediction"),
            "uses_future": da3.get("uses_future"),
            "diagnostic_only": True,
        },
        {
            "schema_version": "stream4d_v100_phase5_variant_row_v1",
            "phase_id": "v100_phase5_da3_d4rt_verifier_audit",
            "variant_id": "G2_F2_plus_D4RT_reliable_anchor",
            "status": "diagnostic_only_not_promoted",
            "source_summary": _rel(V99_D4RT_AH / "summary.json"),
            "provider_available": True,
            "d4rt_contract_pass": d4rt_ah.get("d4rt_contract", {}).get("d4rt_self_overlap_stitch_contract_pass"),
            "real_minus_control_MV_AP_scene": d4rt_ah.get("real_minus_control_MV_AP_scene"),
            "best_real_MV_AP_scene": d4rt_ah.get("best_real_MV_AP_scene"),
            "best_control_MV_AP_scene": d4rt_ah.get("best_control_MV_AP_scene"),
            "control_margin_gate_pass": d4rt_ah.get("control_margin_gate_pass"),
            "diagnostic_only": True,
        },
        {
            "schema_version": "stream4d_v100_phase5_variant_row_v1",
            "phase_id": "v100_phase5_da3_d4rt_verifier_audit",
            "variant_id": "G3_F2_plus_DA3_and_D4RT",
            "status": "diagnostic_only_not_promoted",
            "source_summary": _rel(V99_D4RT_AI / "summary.json"),
            "sim3_alignment_summary": _rel(V99_SIM3 / "summary.json"),
            "sim3_alignment_pass": sim3.get("prefix_causal_alignment_gate_pass"),
            "best_variant_id": d4rt_ai.get("best_variant_id"),
            "best_MV_AP_scene": d4rt_ai.get("best_MV_AP_scene"),
            "best_MV_AP50_scene": d4rt_ai.get("best_MV_AP50_scene"),
            "phase10p_reported_best_MV_AP_scene": d4rt_ai.get("phase10p_reported_best_MV_AP_scene"),
            "metric_gate_pass": d4rt_ai.get("metric_gate_pass"),
            "diagnostic_only": True,
        },
        {
            "schema_version": "stream4d_v100_phase5_variant_row_v1",
            "phase_id": "v100_phase5_da3_d4rt_verifier_audit",
            "variant_id": "G4_F2_plus_DA3_broad_mask_split",
            "status": "blocked_not_run",
            "reason": "No v100 broad-risk DA3 component split implementation exists; not faked.",
            "diagnostic_only": True,
        },
        {
            "schema_version": "stream4d_v100_phase5_variant_row_v1",
            "phase_id": "v100_phase5_da3_d4rt_verifier_audit",
            "variant_id": "G5_F2_plus_dense_semantic_residual",
            "status": "blocked_not_run",
            "reason": "No evidence that dense primitive semantic residual beats mask-level RADIO proxy; not promoted.",
            "diagnostic_only": True,
        },
    ]
    return rows


def main() -> int:
    started = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    phase4 = _load_json(PHASE4_MAIN / "summary.json")
    phase4b = _load_json(PHASE4_REPAIR / "summary.json")
    da3 = _load_json(V99_DA3 / "summary.json")
    d4rt_ah = _load_json(V99_D4RT_AH / "summary.json")
    d4rt_ai = _load_json(V99_D4RT_AI / "summary.json")
    sim3 = _load_json(V99_SIM3 / "summary.json")

    link_rows = _phase4_link_support_rows(PHASE4_MAIN, str(phase4["best_variant_id"]))
    repair_link_rows = _phase4_link_support_rows(PHASE4_REPAIR, str(phase4b["best_variant_id"]))
    all_link_rows = link_rows + repair_link_rows
    supported_count = sum(1 for row in link_rows if bool(row["d4rt_supported"]))
    support_rate = float(supported_count / max(1, len(link_rows)))
    repair_supported_count = sum(1 for row in repair_link_rows if bool(row["d4rt_supported"]))
    repair_support_rate = float(repair_supported_count / max(1, len(repair_link_rows)))

    da3_delta_scene = _num(da3.get("delta_best_real_vs_phase2_scene"))
    d4rt_real_minus_control = _num(d4rt_ah.get("real_minus_control_MV_AP_scene"))
    d4rt_beats_semantic = _num(d4rt_ai.get("best_MV_AP_scene")) >= _num(d4rt_ai.get("phase10p_reported_best_MV_AP_scene")) + 0.006

    gate_rows = [
        {
            "gate_id": "phase4_g0_pass",
            "pass": bool(phase4.get("phase4_pass")),
            "expected": "Phase4 G0 history memory passes",
            "observed": phase4.get("decision"),
            "severity": "input_blocker",
        },
        {
            "gate_id": "da3_increment_scene_ge_0p006",
            "pass": da3_delta_scene >= 0.006,
            "expected": ">=0.006 scene AP delta",
            "observed": da3_delta_scene,
            "severity": "variant_usefulness",
        },
        {
            "gate_id": "d4rt_contract_pass",
            "pass": bool(d4rt_ah.get("d4rt_contract", {}).get("d4rt_self_overlap_stitch_contract_pass")),
            "expected": "D4RT self overlap stitch contract passes",
            "observed": d4rt_ah.get("d4rt_contract", {}).get("d4rt_self_overlap_stitch_contract_pass"),
            "severity": "provider_required",
        },
        {
            "gate_id": "d4rt_real_minus_control_ge_0p005",
            "pass": d4rt_real_minus_control >= 0.005,
            "expected": ">=0.005 scene AP",
            "observed": d4rt_real_minus_control,
            "severity": "control_required",
        },
        {
            "gate_id": "d4rt_beats_semantic_g0_ge_0p006",
            "pass": d4rt_beats_semantic,
            "expected": "D4RT/DA3 combined best >= semantic replay +0.006 scene AP",
            "observed": f"best={d4rt_ai.get('best_MV_AP_scene')} semantic={d4rt_ai.get('phase10p_reported_best_MV_AP_scene')}",
            "severity": "variant_usefulness",
        },
        {
            "gate_id": "phase4_main_accepted_links_d4rt_support_rate_ge_0p5",
            "pass": support_rate >= 0.5,
            "expected": ">=0.5 of Phase4 accepted holdout links have D4RT real_R20/R40 support",
            "observed": f"{supported_count}/{len(link_rows)}={support_rate}",
            "severity": "verifier_required",
        },
        {
            "gate_id": "phase4_repair_accepted_links_d4rt_support_rate_ge_0p5",
            "pass": repair_support_rate >= 0.5,
            "expected": ">=0.5 of Phase4b accepted holdout links have D4RT real_R20/R40 support",
            "observed": f"{repair_supported_count}/{len(repair_link_rows)}={repair_support_rate}",
            "severity": "verifier_required",
        },
        {
            "gate_id": "sim3_alignment_pass_no_future_gt",
            "pass": bool(sim3.get("prefix_causal_alignment_gate_pass")) and not bool(sim3.get("uses_future")) and not bool(sim3.get("uses_gt_for_prediction")),
            "expected": "prefix causal Sim3 alignment pass with no future/GT",
            "observed": f"pass={sim3.get('prefix_causal_alignment_gate_pass')} future={sim3.get('uses_future')} gt={sim3.get('uses_gt_for_prediction')}",
            "severity": "provider_required",
        },
    ]
    failure_rows = [
        {
            "failure_id": row["gate_id"],
            "severity": row["severity"],
            "expected": row["expected"],
            "observed": row["observed"],
            "repair_direction": (
                "If DA3/D4RT verifier fails usefulness/support gates, keep DA3/D4RT diagnostic and do not promote to main method. "
                "Repair would require direct v100 integration of DA3 overlap or D4RT anchor sketches with controls."
            ),
        }
        for row in gate_rows
        if not bool(row["pass"])
    ]
    phase5_pass = not failure_rows

    variant_rows = _variant_rows()
    variant_csv = OUT_DIR / "variant_rows.csv"
    variant_config_csv = OUT_DIR / "variant_config_rows.csv"
    variant_metric_csv = OUT_DIR / "variant_metric_rows.csv"
    link_csv = OUT_DIR / "phase4_link_verifier_rows.csv"
    gate_csv = OUT_DIR / "variant_gate_rows.csv"
    failure_csv = OUT_DIR / "variant_failure_rows.csv"
    performance_csv = OUT_DIR / "performance_rows.csv"
    casebook_csv = OUT_DIR / "casebook_rows.csv"
    artifact_csv = OUT_DIR / "artifact_manifest_rows.csv"

    _write_csv(variant_csv, variant_rows)
    _write_csv(
        variant_config_csv,
        [
            {
                "schema_version": "stream4d_v100_phase5_variant_config_row_v1",
                "phase_id": "v100_phase5_da3_d4rt_verifier_audit",
                "variant_id": row.get("variant_id"),
                "status": row.get("status"),
                "source_summary": row.get("source_summary", ""),
                "sim3_alignment_summary": row.get("sim3_alignment_summary", ""),
                "provider_available": row.get("provider_available", ""),
                "diagnostic_only": row.get("diagnostic_only"),
                "reason": row.get("reason", ""),
            }
            for row in variant_rows
        ],
    )
    _write_csv(
        variant_metric_csv,
        [
            {
                "schema_version": "stream4d_v100_phase5_variant_metric_row_v1",
                "phase_id": "v100_phase5_da3_d4rt_verifier_audit",
                "variant_id": row.get("variant_id"),
                "MV_AP_scene_holdout": row.get("MV_AP_scene_holdout", ""),
                "MV_AP50_scene_holdout": row.get("MV_AP50_scene_holdout", ""),
                "MV_AP_window_holdout": row.get("MV_AP_window_holdout", ""),
                "delta_scene_vs_phase2": row.get("delta_scene_vs_phase2", ""),
                "delta_window_vs_phase2": row.get("delta_window_vs_phase2", ""),
                "real_minus_control_MV_AP_scene": row.get("real_minus_control_MV_AP_scene", ""),
                "best_real_MV_AP_scene": row.get("best_real_MV_AP_scene", ""),
                "best_control_MV_AP_scene": row.get("best_control_MV_AP_scene", ""),
                "best_MV_AP_scene": row.get("best_MV_AP_scene", ""),
                "best_MV_AP50_scene": row.get("best_MV_AP50_scene", ""),
                "semantic_reference_MV_AP_scene": row.get("phase10p_reported_best_MV_AP_scene", ""),
                "phase4_pass": row.get("phase4_pass", ""),
                "metric_gate_pass": row.get("metric_gate_pass", ""),
                "control_margin_gate_pass": row.get("control_margin_gate_pass", ""),
                "uses_gt_for_prediction": row.get("uses_gt_for_prediction", ""),
                "uses_future": row.get("uses_future", ""),
            }
            for row in variant_rows
        ],
    )
    _write_csv(link_csv, all_link_rows)
    _write_csv(gate_csv, gate_rows)
    _write_csv(failure_csv, failure_rows)
    _write_csv(
        performance_csv,
        [
            {
                "schema_version": "stream4d_v100_phase5_performance_row_v1",
                "phase_id": "v100_phase5_da3_d4rt_verifier_audit",
                "case_id": "provider_control_and_phase4_link_support_audit",
                "runtime_sec": time.time() - started,
                "v65_evaluator_runs": 0,
                "phase4_main_link_rows_checked": len(link_rows),
                "phase4_repair_link_rows_checked": len(repair_link_rows),
                "gpu_used": False,
            }
        ],
    )
    _write_csv(
        casebook_csv,
        [
            {
                "schema_version": "stream4d_v100_phase5_casebook_row_v1",
                "phase_id": "v100_phase5_da3_d4rt_verifier_audit",
                "case_id": "phase4_g0_input_blocker",
                "evidence": f"phase4_decision={phase4.get('decision')} phase4b_decision={phase4b.get('decision')}",
                "interpretation": "G0 semantic/mask-view history memory did not pass Phase4 gates, so DA3/D4RT cannot be promoted on top of a passing history-memory baseline.",
            },
            {
                "schema_version": "stream4d_v100_phase5_casebook_row_v1",
                "phase_id": "v100_phase5_da3_d4rt_verifier_audit",
                "case_id": "da3_increment_too_small",
                "evidence": f"delta_scene_vs_phase2={da3_delta_scene}",
                "interpretation": "DA3 link-verifier evidence is below the v100 +0.006 scene AP usefulness threshold.",
            },
            {
                "schema_version": "stream4d_v100_phase5_casebook_row_v1",
                "phase_id": "v100_phase5_da3_d4rt_verifier_audit",
                "case_id": "d4rt_control_positive_but_not_method_positive",
                "evidence": f"real_minus_control={d4rt_real_minus_control} best={d4rt_ai.get('best_MV_AP_scene')} semantic_reference={d4rt_ai.get('phase10p_reported_best_MV_AP_scene')}",
                "interpretation": "D4RT has a real-vs-control signal, but the combined DA3/D4RT path does not beat the semantic reference by the required margin.",
            },
            {
                "schema_version": "stream4d_v100_phase5_casebook_row_v1",
                "phase_id": "v100_phase5_da3_d4rt_verifier_audit",
                "case_id": "phase4_links_lack_d4rt_support",
                "evidence": f"main_supported={supported_count}/{len(link_rows)} repair_supported={repair_supported_count}/{len(repair_link_rows)}",
                "interpretation": "Most accepted semantic history links have no high-overlap real_R20/R40 D4RT anchor support, so D4RT cannot validate the current semantic linker.",
            },
        ],
    )
    artifacts = [
        (variant_csv, "csv", "Phase5 G0-G5 status rows"),
        (variant_config_csv, "csv", "Phase5 G0-G5 configuration/status rows"),
        (variant_metric_csv, "csv", "Phase5 G0-G5 metric rows"),
        (link_csv, "csv", "D4RT support audit for Phase4 accepted semantic links"),
        (gate_csv, "csv", "Phase5 gates"),
        (failure_csv, "csv", "Phase5 failures"),
        (performance_csv, "csv", "Phase5 audit runtime"),
        (casebook_csv, "csv", "Phase5 evidence casebook"),
    ]
    _write_csv(
        artifact_csv,
        [
            {
                "schema_version": "stream4d_v100_phase5_artifact_manifest_row_v1",
                "phase_id": "v100_phase5_da3_d4rt_verifier_audit",
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
        "schema_version": "stream4d_v100_phase5_da3_d4rt_verifier_audit_summary_v1",
        "phase_id": "v100_phase5_da3_d4rt_verifier_audit",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "runtime_sec": time.time() - started,
        "decision": "PASS_ENTER_PHASE6" if phase5_pass else "NO_GO_DA3_D4RT_VERIFIER_DIAGNOSTIC_ONLY",
        "phase5_pass": phase5_pass,
        "failure_count": len(failure_rows),
        "phase4_main_best_variant": phase4.get("best_variant_id"),
        "phase4_main_holdout_MV_AP_scene": phase4.get("best_holdout_MV_AP_scene"),
        "phase4_repair_best_variant": phase4b.get("best_variant_id"),
        "phase4_repair_holdout_MV_AP_scene": phase4b.get("best_holdout_MV_AP_scene"),
        "da3_delta_scene_vs_phase2": da3_delta_scene,
        "d4rt_real_minus_control_MV_AP_scene": d4rt_real_minus_control,
        "d4rt_ai_best_MV_AP_scene": d4rt_ai.get("best_MV_AP_scene"),
        "d4rt_ai_semantic_reference_MV_AP_scene": d4rt_ai.get("phase10p_reported_best_MV_AP_scene"),
        "phase4_main_d4rt_supported_link_count": supported_count,
        "phase4_main_accepted_holdout_link_count": len(link_rows),
        "phase4_main_d4rt_support_rate": support_rate,
        "phase4_repair_d4rt_supported_link_count": repair_supported_count,
        "phase4_repair_accepted_holdout_link_count": len(repair_link_rows),
        "phase4_repair_d4rt_support_rate": repair_support_rate,
        "sim3_alignment_pass": sim3.get("prefix_causal_alignment_gate_pass"),
        "formal_claim_allowed": False,
        "outputs": {
            "summary": _rel(OUT_DIR / "summary.json"),
            "variant_rows": _rel(variant_csv),
            "variant_config_rows": _rel(variant_config_csv),
            "variant_metric_rows": _rel(variant_metric_csv),
            "phase4_link_verifier_rows": _rel(link_csv),
            "variant_gate_rows": _rel(gate_csv),
            "variant_failure_rows": _rel(failure_csv),
            "performance_rows": _rel(performance_csv),
            "casebook_rows": _rel(casebook_csv),
            "artifact_manifest_rows": _rel(artifact_csv),
        },
    }
    _write_json(OUT_DIR / "summary.json", summary)
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))
    return 0 if phase5_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
