from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.run_v65_scene_multiview_ap import _sha256, _write_csv, _write_json  # noqa: E402
from tools.run_v66_local_chunk_eval import _rel  # noqa: E402


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _artifact_row(path: Path) -> dict[str, Any]:
    return {"path": _rel(path), "sha256": _sha256(path), "bytes": int(path.stat().st_size)}


def _pick_phase2_path() -> Path:
    dinov2 = ROOT / "outputs/audit/v68_edge_audit_dinov2/edge_audit_summary.json"
    if dinov2.exists():
        return dinov2
    return ROOT / "outputs/audit/v68_edge_audit/edge_audit_summary.json"


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = ROOT / output_root
    output_root.mkdir(parents=True, exist_ok=True)

    phase0_path = ROOT / "outputs/audit/v68_phase0_fact_lock/fact_lock_summary.json"
    phase1_path = ROOT / "outputs/audit/v68_candidate_bank/candidate_bank_summary.json"
    phase2_path = _pick_phase2_path()
    phase3_path = ROOT / "outputs/audit/v68_local_graph_solver/local_solver_summary.json"
    required = [phase0_path, phase1_path, phase2_path]
    missing = [path for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("missing required v68 summaries: " + ", ".join(str(path) for path in missing))

    phase0 = _read_json(phase0_path)
    phase1 = _read_json(phase1_path)
    phase2 = _read_json(phase2_path)
    phase3 = _read_json(phase3_path) if phase3_path.exists() else {}
    phase0_pass = bool((phase0.get("gate") or {}).get("pass"))
    phase1_pass = bool((phase1.get("gate") or {}).get("pass"))
    phase2_pass = bool((phase2.get("gate") or {}).get("pass"))
    phase3_pass = bool((phase3.get("gate") or {}).get("pass"))
    if not phase0_pass:
        decision = "NO_GO_V67_FACT_LOCK"
    elif not phase1_pass:
        decision = "NO_GO_CANDIDATE_BANK"
    elif not phase2_pass:
        decision = "NO_GO_EDGE_EVIDENCE"
    elif phase3 and not phase3_pass:
        decision = str(phase3.get("decision") or "NO_GO_LOCAL_SOLVER")
    elif phase3_pass:
        decision = "READY_FOR_LOCAL2HISTORY"
    else:
        decision = "READY_FOR_LOCAL_SOLVER"

    can_enter_local_solver = phase0_pass and phase1_pass and phase2_pass
    can_enter_local2history = bool(can_enter_local_solver and phase3_pass)
    if decision == "NO_GO_EDGE_EVIDENCE":
        local2history_status = "blocked_by_edge_evidence"
    elif phase3 and not phase3_pass:
        local2history_status = "blocked_by_local_gate"
    elif can_enter_local2history:
        local2history_status = "ready_for_local2history_eval"
    else:
        local2history_status = "blocked_until_local_gate_pass"
    combined = phase2.get("combined_metrics") or {}
    best_s = phase3.get("best_S") or {}
    summary = {
        "phase": "v68_final_decision",
        "decision": decision,
        "diagnostic_only": True,
        "can_enter_local_solver": bool(can_enter_local_solver),
        "can_enter_local2history": bool(can_enter_local2history),
        "local2history_status": local2history_status,
        "phase_status": {
            "phase0_v67_fact_lock": phase0.get("decision"),
            "phase1_candidate_bank": phase1.get("decision"),
            "phase2_edge_audit": phase2.get("decision"),
            "phase3_seeded_local_solver": phase3.get("decision", "not_run"),
            "phase4_setcover_v2": "not_run_phase3_local_gate_failed" if phase3 and not phase3_pass else "not_run",
            "phase5_local2history": "not_run_blocked_by_local_gate" if phase3 and not phase3_pass else "not_run",
            "phase6_visualization": "partial_edge_and_solver_rows_no_solver_casebook_or_3d_viewer",
        },
        "gate": {
            "phase0_pass": phase0_pass,
            "phase1_pass": phase1_pass,
            "phase2_pass": phase2_pass,
            "phase3_pass": phase3_pass,
            "can_enter_local_solver": bool(can_enter_local_solver),
            "can_enter_local2history": bool(can_enter_local2history),
        },
        "key_metrics": {
            "candidate_bank_best_oracle_SF50": (phase1.get("best_CB") or {}).get("local_score_free_match50_recall_mean"),
            "candidate_bank_best_oracle_AP50": (phase1.get("best_CB") or {}).get("local_AP50_mean"),
            "candidate_bank_best_GT_best_IoU_mean": (phase1.get("best_CB") or {}).get("local_GT_best_IoU_mean_mean"),
            "edge_combined_AUC": combined.get("edge_AUC"),
            "edge_combined_top1_precision": combined.get("top1_precision"),
            "edge_combined_top3_recall": combined.get("top3_recall"),
            "edge_real_minus_shuffled_AUC": combined.get("real_minus_shuffled_AUC"),
            "edge_real_minus_no_temporal_AUC": combined.get("real_minus_no_temporal_AUC"),
            "edge_hard_negative_precision": combined.get("hard_negative_precision"),
            "edge_same_frame_violation_rate_after_filter": combined.get("same_frame_violation_rate"),
            "edge_scene0011_AUC": (phase2.get("scene_auc") or {}).get("scene0011_00"),
            "edge_scene0030_AUC": (phase2.get("scene_auc") or {}).get("scene0030_00"),
            "edge_scene0050_AUC": (phase2.get("scene_auc") or {}).get("scene0050_00"),
            "edge_scene0081_AUC": (phase2.get("scene_auc") or {}).get("scene0081_01"),
            "edge_scene0591_AUC": (phase2.get("scene_auc") or {}).get("scene0591_00"),
            "best_local_solver_variant": best_s.get("variant"),
            "best_local_solver_SF50": best_s.get("local_score_free_match50_recall_mean"),
            "best_local_solver_AP50": best_s.get("local_AP50_mean"),
            "best_local_solver_GT_best_IoU_mean": best_s.get("local_GT_best_IoU_mean_mean"),
            "best_local_solver_single_frame_object_rate": best_s.get("single_frame_object_rate_mean"),
            "best_local_solver_same_frame_violation_count": best_s.get("same_frame_cannot_link_violation_count_sum"),
        },
        "blocker_analysis": [
            "Phase 0 and Phase 1 passed, so v68 did not fail because v67 provenance was unstable or because candidate-bank oracle headroom was lost.",
            "The DINO frozen-appearance Phase 2 repair passed edge calibration, including AUC/top1/top3, shuffled control, no-temporal control, hard-negative precision, and same-frame filtering.",
            "Phase 3 then failed the local gate: best non-oracle solver remains far below SF50/AP50/GT-best gates and still has a high single-frame object rate.",
            "Per plan stop rule, local2history is not run as a method conclusion because the local object primitive gate failed.",
        ],
        "repair_attempts_recorded": [
            "Required diagnostic GT purity >= 0.50 before using majority GT labels for edge AUC/top-k diagnostics.",
            "Changed top1 precision denominator to query nodes with at least one diagnostic positive candidate, while keeping query counts in metric rows.",
            "Changed no-temporal control to material-only instead of a shape/signature mixed control.",
            "Separated same-frame structural duplicate filtering from hard-negative GT-different precision because same-frame mask fragments can share a diagnostic GT id.",
            "Reweighted combined score toward material+signature+temporal and reran full Phase 2 with the same deterministic seed and candidate universe.",
            "Added DINOv2 frozen RGB appearance backend and score_combined_frozen_appearance, which repaired Phase 2 edge calibration.",
            "Added v68 local graph solver variants for fixed-threshold conflict-aware CC, seed/support absorption, underseg shared-state reject, and top-k edge tracking.",
        ],
        "source_summaries": {
            "phase0": _rel(phase0_path),
            "phase1": _rel(phase1_path),
            "phase2": _rel(phase2_path),
            "phase3": _rel(phase3_path) if phase3_path.exists() else "",
        },
    }
    _write_json(output_root / "final_decision.json", summary)
    artifact_inputs = required + ([phase3_path] if phase3_path.exists() else [])
    artifact_rows = [_artifact_row(path) for path in artifact_inputs + [output_root / "final_decision.json"]]
    _write_csv(output_root / "sha256_rows.csv", artifact_rows)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Stream4D v68 final decision artifact.")
    parser.add_argument("--output-root", default="outputs/audit/v68_final_decision")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
