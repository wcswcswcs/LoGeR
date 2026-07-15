#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


STREAM3D_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = STREAM3D_ROOT.parent
AUDIT_ROOT = STREAM3D_ROOT / "outputs" / "audit"
PHASE_ID = "v103_supp_r6_final_decision"
DEFAULT_OUT = AUDIT_ROOT / "v103_supp_r6_final_decision"
DEFAULT_R6_0_ROOT = AUDIT_ROOT / "v103_supp_r6_phase0_fact_lock"
DEFAULT_R6_1_ROOT = AUDIT_ROOT / "v103_supp_r6_phase1_edge_attribution_casebook"
DEFAULT_R6_2_FEATURE_ROOT = AUDIT_ROOT / "v103_supp_r6_phase2_support_conditioned_feature"
DEFAULT_R6_2_LOCAL_AP_ROOT = AUDIT_ROOT / "v103_supp_r6_phase2_support_conditioned_local_ap"
DEFAULT_R6_5_ROOT = AUDIT_ROOT / "v103_supp_r6_phase5_support_ranking_extent"
DEFAULT_R6_6_ROOT = AUDIT_ROOT / "v103_supp_r6_phase6_gt_coverage_inconsistency"
D9_VARIANT = "D9_affinity_merge_tau065_top1_broad_support_veto"


def _project(path: Path | str) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    if p.parts and p.parts[0] == "Stream3D":
        return REPO_ROOT / p
    return STREAM3D_ROOT / p


def _rel(path: Path | str) -> str:
    p = Path(path)
    try:
        return p.resolve().relative_to(REPO_ROOT).as_posix()
    except Exception:
        return p.as_posix()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return _rel(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


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
            writer.writerow({key: _jsonable(row.get(key, "")) for key in fields})


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _read_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _num(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        out = float(value)
        if not np.isfinite(out):
            return default
        return out
    except Exception:
        return default


def _gate(gate_id: str, passed: bool, observed: Any, required: Any, detail: str = "") -> dict[str, Any]:
    return {
        "schema_version": "stream4d_v103_supp_r6_final_gate_row_v1",
        "phase_id": PHASE_ID,
        "gate_id": gate_id,
        "pass": bool(passed),
        "observed": observed,
        "required": required,
        "detail": detail,
        "uses_gt_for_prediction": False,
        "uses_future": False,
    }


def _best_r6f_d9(local_rows: list[dict[str, Any]]) -> dict[str, Any]:
    d9_rows = [r for r in local_rows if str(r.get("phase6d_variant_id")) == D9_VARIANT]
    if not d9_rows:
        return {}
    return max(d9_rows, key=lambda r: _num(r.get("MV_AP_window"), -1.0))


def _best_r6sr(metric_rows: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [r for r in metric_rows if str(r.get("variant_id")) != "R6SR0_current_d9_score_replay"]
    if not rows:
        return {}
    return max(rows, key=lambda r: _num(r.get("MV_AP_window"), -1.0))


def build(args: argparse.Namespace) -> dict[str, Any]:
    t0 = time.time()
    out = _project(args.output_root)
    if out.exists() and any(out.iterdir()) and not args.force:
        raise SystemExit(f"output root already exists and is non-empty; pass --force: {_rel(out)}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "last_command.txt").write_text(" ".join([sys.executable, str(Path(__file__)), *sys.argv[1:]]) + "\n", encoding="utf-8")

    r6_0_root = _project(args.r6_0_root)
    r6_1_root = _project(args.r6_1_root)
    r6_2_feature_root = _project(args.r6_2_feature_root)
    r6_2_local_root = _project(args.r6_2_local_ap_root)
    r6_5_root = _project(args.r6_5_root)
    r6_6_root = _project(args.r6_6_root)

    s0 = _read_json(r6_0_root / "summary.json")
    s1 = _read_json(r6_1_root / "summary.json")
    s2f = _read_json(r6_2_feature_root / "summary.json")
    s2 = _read_json(r6_2_local_root / "summary.json")
    s5 = _read_json(r6_5_root / "summary.json")
    s6 = _read_json(r6_6_root / "summary.json")
    r6_2_metrics = _read_rows(r6_2_local_root / "variant_metric_rows.csv")
    r6_5_metrics = _read_rows(r6_5_root / "variant_metric_rows.csv")
    r6_5_compare = _read_rows(r6_5_root / "control_comparison_rows.csv")
    r6_6_cov = _read_rows(r6_6_root / "gt_object_coverage_summary_rows.csv")
    r6_6_incon = _read_rows(r6_6_root / "three_d_inconsistency_summary_rows.csv")

    best_r6f = _best_r6f_d9(r6_2_metrics)
    best_r6sr = _best_r6sr(r6_5_metrics)
    best_r6sr_compare = next((r for r in r6_5_compare if str(r.get("variant_id")) == str(best_r6sr.get("variant_id", ""))), {})
    best_real_variant_id = str(best_r6sr.get("variant_id") or best_r6f.get("r5_feature_variant_id", ""))
    best_real_mv_ap = _num(best_r6sr.get("MV_AP_window", best_r6f.get("MV_AP_window", 0.0)))
    best_real_ap50 = _num(best_r6sr.get("MV_AP50_window", best_r6f.get("MV_AP50_window", 0.0)))
    replay_mv_ap = _num(s0.get("current_replay_MV_AP_window"), 0.0)
    replay_ap50 = _num(s0.get("current_replay_MV_AP50_window"), 0.0)
    best_control_mv_ap = max(
        _num(best_r6sr_compare.get("shuffled_MV_AP_window"), 0.0),
        _num(best_r6sr_compare.get("stale_proxy_MV_AP_window"), 0.0),
    )
    best_real_minus_best_control = best_real_mv_ap - best_control_mv_ap if best_r6sr_compare else 0.0

    exact_stale_available = bool(s5.get("stale_support_control_exact_available", False))
    subset_gate_pass = bool(s2.get("phase_r6_2_local_ap_pass", False) or s5.get("phase_r6_5_pass", False))
    full_dev_allowed = False
    holdout_allowed = False
    history_allowed = False

    final_gate_rows = [
        _gate("R6_0_fact_lock_pass", bool(s0.get("phase_r6_0_pass", False)), s0.get("decision", ""), "PASS"),
        _gate("R6_1_edge_casebook_complete", bool(s1.get("phase_r6_1_complete", False)), s1.get("decision", ""), "complete"),
        _gate("R6_2_feature_builder_pass", bool(s2f.get("phase_r6_2_feature_pass", False)), s2f.get("decision", ""), "PASS"),
        _gate("R6_2_support_conditioned_local_ap_pass", bool(s2.get("phase_r6_2_local_ap_pass", False)), s2.get("decision", ""), "PASS"),
        _gate("R6_5_support_ranking_extent_pass", bool(s5.get("phase_r6_5_pass", False)), s5.get("decision", ""), "PASS"),
        _gate("R6_6_gt_coverage_inconsistency_diag_complete", bool(s6.get("phase_r6_6_diag_complete", False)), s6.get("decision", ""), "complete"),
        _gate("R6SR_best_real_minus_best_control_ge_0p003", best_real_minus_best_control >= 0.003, best_real_minus_best_control, 0.003),
        _gate("exact_stale_support_control_available", exact_stale_available, exact_stale_available, True),
        _gate("subset_gate_pass", subset_gate_pass, subset_gate_pass, True),
        _gate("full_dev_allowed", full_dev_allowed, full_dev_allowed, True, "blocked because subset gate did not pass"),
        _gate("holdout_allowed", holdout_allowed, holdout_allowed, True, "blocked because full-dev is not allowed"),
        _gate("history_allowed", history_allowed, history_allowed, True, "blocked because local full-dev did not pass"),
    ]

    best_variant_metric_rows = []
    if best_r6f:
        best_variant_metric_rows.append(
            {
                "schema_version": "stream4d_v103_supp_r6_best_variant_metric_row_v1",
                "phase_id": PHASE_ID,
                "source_phase": "R6_2_support_conditioned_feature_membership_D9",
                "variant_id": best_r6f.get("r5_feature_variant_id", ""),
                "phase6d_variant_id": best_r6f.get("phase6d_variant_id", ""),
                "MV_AP_window": best_r6f.get("MV_AP_window", ""),
                "MV_AP50_window": best_r6f.get("MV_AP50_window", ""),
                "ScoreFreeMatch50_window": best_r6f.get("ScoreFreeMatch50_window", ""),
                "same_frame_collision_count": best_r6f.get("same_frame_collision_count", ""),
                "pixel_collision_rate": best_r6f.get("pixel_collision_rate", ""),
                "missing_mask_raster_count": best_r6f.get("missing_mask_raster_count", ""),
                "promotion_status": "NO_GO_R6_2_LOCAL_AP_GATE_FAILED",
            }
        )
    if best_r6sr:
        best_variant_metric_rows.append(
            {
                "schema_version": "stream4d_v103_supp_r6_best_variant_metric_row_v1",
                "phase_id": PHASE_ID,
                "source_phase": "R6_5_support_ranking_extent",
                "variant_id": best_r6sr.get("variant_id", ""),
                "MV_AP_window": best_r6sr.get("MV_AP_window", ""),
                "MV_AP50_window": best_r6sr.get("MV_AP50_window", ""),
                "ScoreFreeMatch50_window": best_r6sr.get("ScoreFreeMatch50_window", ""),
                "real_minus_shuffled_score_repair": best_r6sr_compare.get("real_minus_shuffled_score_repair", ""),
                "real_minus_stale_proxy_MV_AP_window": best_r6sr_compare.get("real_minus_stale_proxy_MV_AP_window", ""),
                "best_real_minus_best_control_MV_AP_window": best_real_minus_best_control,
                "promotion_status": "NO_GO_R6_5_CONTROL_GATE_FAILED",
            }
        )

    support_attribution_summary_rows = [
        {
            "schema_version": "stream4d_v103_supp_r6_support_attribution_summary_row_v1",
            "phase_id": PHASE_ID,
            "source_phase": "R6_1_edge_attribution_casebook",
            "support_contribution_mode": s1.get("support_attribution", {}).get("support_contribution_mode", ""),
            "support_has_attributable_signal": s1.get("support_attribution", {}).get("support_has_attributable_signal", ""),
            "support_improves_over_anchor_only": s1.get("support_attribution", {}).get("support_improves_over_anchor_only", ""),
            "support_feature_hurts_replay": s1.get("support_attribution", {}).get("support_feature_hurts_replay", ""),
            "false_bridge_risk_nonzero": s1.get("support_attribution", {}).get("false_bridge_risk_nonzero", ""),
            "accepted_diff_gt_edge_count": s1.get("accepted_diff_gt_edge_count", ""),
            "exact_leave_one_family_missing_count": s1.get("exact_leave_one_family_missing_count", ""),
            "decision": s1.get("decision", ""),
        },
        {
            "schema_version": "stream4d_v103_supp_r6_support_attribution_summary_row_v1",
            "phase_id": PHASE_ID,
            "source_phase": "R6_2_feature_and_local_ap",
            "support_feature_validity_signal": "R6F1/R6F2 feature construction passed, but D9 membership AP stayed below replay and local AP gate failed",
            "decision": s2.get("decision", ""),
            "fully_passing_r6_feature_variants": ";".join(map(str, s2.get("fully_passing_r6_feature_variants", []))),
        },
        {
            "schema_version": "stream4d_v103_supp_r6_support_attribution_summary_row_v1",
            "phase_id": PHASE_ID,
            "source_phase": "R6_5_support_ranking_extent",
            "support_ranking_signal": "small MV_AP movement exists but does not beat shuffled/best control by 0.003; exact stale control unavailable",
            "decision": s5.get("decision", ""),
            "best_real_variant_id": best_real_variant_id,
            "best_real_minus_best_control_MV_AP_window": best_real_minus_best_control,
        },
    ]

    codex_attempt_rows = [
        {
            "schema_version": "stream4d_v103_supp_r6_codex_attempt_row_v1",
            "phase_id": PHASE_ID,
            "attempt_id": "R6_0_fact_lock",
            "artifact_root": _rel(r6_0_root),
            "status": s0.get("decision", ""),
            "runs_AP": False,
            "uses_gt_for_prediction": False,
        },
        {
            "schema_version": "stream4d_v103_supp_r6_codex_attempt_row_v1",
            "phase_id": PHASE_ID,
            "attempt_id": "R6_1_edge_attribution_casebook",
            "artifact_root": _rel(r6_1_root),
            "status": s1.get("decision", ""),
            "runs_AP": False,
            "uses_gt_for_prediction": False,
        },
        {
            "schema_version": "stream4d_v103_supp_r6_codex_attempt_row_v1",
            "phase_id": PHASE_ID,
            "attempt_id": "R6_2_support_conditioned_feature",
            "artifact_root": _rel(r6_2_feature_root),
            "status": s2f.get("decision", ""),
            "runs_AP": False,
            "uses_gt_for_prediction": False,
        },
        {
            "schema_version": "stream4d_v103_supp_r6_codex_attempt_row_v1",
            "phase_id": PHASE_ID,
            "attempt_id": "R6_2_support_conditioned_local_ap",
            "artifact_root": _rel(r6_2_local_root),
            "status": s2.get("decision", ""),
            "runs_AP": True,
            "uses_gt_for_prediction": False,
        },
        {
            "schema_version": "stream4d_v103_supp_r6_codex_attempt_row_v1",
            "phase_id": PHASE_ID,
            "attempt_id": "R6_5_support_ranking_extent",
            "artifact_root": _rel(r6_5_root),
            "status": s5.get("decision", ""),
            "runs_AP": True,
            "uses_gt_for_prediction": False,
        },
        {
            "schema_version": "stream4d_v103_supp_r6_codex_attempt_row_v1",
            "phase_id": PHASE_ID,
            "attempt_id": "R6_6_gt_coverage_3d_inconsistency_diagnostic",
            "artifact_root": _rel(r6_6_root),
            "status": s6.get("decision", ""),
            "runs_AP": False,
            "uses_gt_for_prediction": False,
        },
        {
            "schema_version": "stream4d_v103_supp_r6_codex_attempt_row_v1",
            "phase_id": PHASE_ID,
            "attempt_id": "R6_3_R6_4_exact_anchor_skeleton_propagation",
            "artifact_root": "",
            "status": "NOT_RUN_IN_THIS_BRANCH_EXACT_FAMILY_NOT_CLAIMED; existing direct-pair/score rows treated only as proxy hints",
            "runs_AP": False,
            "uses_gt_for_prediction": False,
        },
    ]

    _write_csv(out / "final_gate_rows.csv", final_gate_rows)
    _write_csv(out / "best_variant_metric_rows.csv", best_variant_metric_rows)
    _write_csv(out / "support_attribution_summary_rows.csv", support_attribution_summary_rows)
    _write_csv(out / "gt_coverage_summary_rows.csv", r6_6_cov)
    _write_csv(out / "three_d_inconsistency_summary_rows.csv", r6_6_incon)
    _write_csv(out / "control_comparison_rows.csv", r6_5_compare)
    _write_csv(out / "codex_attempt_rows.csv", codex_attempt_rows)

    artifact_rows = [
        {
            "schema_version": "stream4d_v103_supp_r6_final_artifact_row_v1",
            "phase_id": PHASE_ID,
            "artifact_role": role,
            "path": _rel(path),
            "exists": path.exists(),
            "size_bytes": path.stat().st_size if path.exists() else 0,
            "uses_gt_for_prediction": False,
            "uses_future": False,
        }
        for role, path in [
            ("summary", out / "summary.json"),
            ("final_gate_rows", out / "final_gate_rows.csv"),
            ("best_variant_metric_rows", out / "best_variant_metric_rows.csv"),
            ("support_attribution_summary_rows", out / "support_attribution_summary_rows.csv"),
            ("gt_coverage_summary_rows", out / "gt_coverage_summary_rows.csv"),
            ("three_d_inconsistency_summary_rows", out / "three_d_inconsistency_summary_rows.csv"),
            ("control_comparison_rows", out / "control_comparison_rows.csv"),
            ("codex_attempt_rows", out / "codex_attempt_rows.csv"),
            ("last_command", out / "last_command.txt"),
        ]
    ]
    _write_csv(out / "artifact_rows.csv", artifact_rows)

    summary = {
        "schema_version": "stream4d_v103_supp_r6_final_decision_summary_v1",
        "phase_id": PHASE_ID,
        "decision": "NO_GO_R6_SUPPORT_NOT_YET_OBJECT_SPECIFIC",
        "best_real_variant_id": best_real_variant_id,
        "best_real_MV_AP_window": best_real_mv_ap,
        "best_real_MV_AP50_window": best_real_ap50,
        "best_real_minus_replay_MV_AP_window": best_real_mv_ap - replay_mv_ap,
        "best_real_minus_best_control_MV_AP_window": best_real_minus_best_control,
        "support_used_as_feature": True,
        "support_used_as_anchor_confirmation": False,
        "support_used_as_skeleton_confirmation": False,
        "support_used_as_propagation": False,
        "support_used_as_ranking": True,
        "full_dev_allowed": full_dev_allowed,
        "holdout_allowed": holdout_allowed,
        "history_allowed": history_allowed,
        "primary_blocker": "R6_2_membership_gate_failed_and_R6_5_score_repair_did_not_beat_controls",
        "secondary_blockers": [
            "support_false_bridge_diagnostic_nonzero",
            "real_minus_shuffled_score_repair_lt_0p003",
            "exact_stale_support_control_unavailable",
            "exact_R6_3_R6_4_anchor_skeleton_propagation_family_not_claimed",
        ],
        "failure_count": int(sum(1 for row in final_gate_rows if not bool(row["pass"]))),
        "runs_AP": False,
        "uses_gt_for_prediction": False,
        "uses_gt_for_eval": True,
        "uses_future": False,
        "runtime_sec": time.time() - t0,
        "outputs": {
            "summary": _rel(out / "summary.json"),
            "final_gate_rows": _rel(out / "final_gate_rows.csv"),
            "best_variant_metric_rows": _rel(out / "best_variant_metric_rows.csv"),
            "support_attribution_summary_rows": _rel(out / "support_attribution_summary_rows.csv"),
            "gt_coverage_summary_rows": _rel(out / "gt_coverage_summary_rows.csv"),
            "three_d_inconsistency_summary_rows": _rel(out / "three_d_inconsistency_summary_rows.csv"),
            "control_comparison_rows": _rel(out / "control_comparison_rows.csv"),
            "codex_attempt_rows": _rel(out / "codex_attempt_rows.csv"),
            "artifact_rows": _rel(out / "artifact_rows.csv"),
        },
        "truthfulness_note": (
            "R6 final decision is a No-Go. It does not authorize full-dev, holdout, or history. "
            "R6-3/R6-4 exact anchor/skeleton support confirmation was not claimed as completed; existing D11/D13 rows are proxy hints only."
        ),
    }
    _write_json(out / "summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Stream4D v103 R6 final decision collector.")
    parser.add_argument("--output-root", default=str(DEFAULT_OUT))
    parser.add_argument("--r6-0-root", default=str(DEFAULT_R6_0_ROOT))
    parser.add_argument("--r6-1-root", default=str(DEFAULT_R6_1_ROOT))
    parser.add_argument("--r6-2-feature-root", default=str(DEFAULT_R6_2_FEATURE_ROOT))
    parser.add_argument("--r6-2-local-ap-root", default=str(DEFAULT_R6_2_LOCAL_AP_ROOT))
    parser.add_argument("--r6-5-root", default=str(DEFAULT_R6_5_ROOT))
    parser.add_argument("--r6-6-root", default=str(DEFAULT_R6_6_ROOT))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    summary = build(args)
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
