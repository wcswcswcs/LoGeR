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
from tools.run_v66_local_chunk_eval import _float_or_none, _rel  # noqa: E402


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _rooted(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else ROOT / path


def _metric_row(name: str, value: Any, threshold: str, passed: bool, source: str) -> dict[str, Any]:
    return {
        "metric": name,
        "value": value,
        "threshold": threshold,
        "pass": bool(passed),
        "source": source,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_root = _rooted(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    final_path = _rooted(args.v68_final_decision)
    candidate_path = _rooted(args.v68_candidate_summary)
    edge_path = _rooted(args.v68_edge_summary)
    solver_path = _rooted(args.v68_local_solver_summary)

    final = _load_json(final_path)
    candidate = _load_json(candidate_path)
    edge = _load_json(edge_path)
    solver = _load_json(solver_path)

    best_cb = candidate.get("best_CB") or {}
    edge_metrics = edge.get("combined_metrics") or {}
    best_s = solver.get("best_S") or {}

    v68_decision = final.get("decision")
    candidate_sf50 = _float_or_none(best_cb.get("local_score_free_match50_recall_mean"))
    candidate_ap50 = _float_or_none(best_cb.get("local_AP50_mean"))
    candidate_gt_best = _float_or_none(best_cb.get("local_GT_best_IoU_mean_mean"))
    edge_auc = _float_or_none(edge_metrics.get("edge_AUC"))
    edge_top1 = _float_or_none(edge_metrics.get("top1_precision"))
    edge_top3 = _float_or_none(edge_metrics.get("top3_recall"))
    edge_real_minus_shuffled = _float_or_none(edge_metrics.get("real_minus_shuffled_AUC"))
    edge_real_minus_no_temporal = _float_or_none(edge_metrics.get("real_minus_no_temporal_AUC"))
    solver_sf50 = _float_or_none(best_s.get("local_score_free_match50_recall_mean"))
    solver_ap50 = _float_or_none(best_s.get("local_AP50_mean"))
    solver_gt_best = _float_or_none(best_s.get("local_GT_best_IoU_mean_mean"))
    solver_single_frame = _float_or_none(best_s.get("single_frame_object_rate_mean"))
    solver_violations = int(float(best_s.get("same_frame_cannot_link_violation_count_sum") or 0))
    can_enter_local2history = bool(final.get("can_enter_local2history"))

    metric_rows = [
        _metric_row("v68_decision", v68_decision, "NO_GO_OVERFRAGMENT", v68_decision == "NO_GO_OVERFRAGMENT", _rel(final_path)),
        _metric_row(
            "candidate_bank_oracle_SF50",
            candidate_sf50,
            ">=0.50",
            candidate_sf50 is not None and candidate_sf50 >= 0.50,
            _rel(candidate_path),
        ),
        _metric_row(
            "candidate_bank_oracle_AP50",
            candidate_ap50,
            "record_only",
            candidate_ap50 is not None,
            _rel(candidate_path),
        ),
        _metric_row(
            "candidate_bank_GT_best_IoU",
            candidate_gt_best,
            "record_only",
            candidate_gt_best is not None,
            _rel(candidate_path),
        ),
        _metric_row("DINO_edge_AUC", edge_auc, ">=0.80", edge_auc is not None and edge_auc >= 0.80, _rel(edge_path)),
        _metric_row(
            "DINO_edge_top1_precision",
            edge_top1,
            ">=0.85",
            edge_top1 is not None and edge_top1 >= 0.85,
            _rel(edge_path),
        ),
        _metric_row(
            "DINO_edge_top3_recall",
            edge_top3,
            "record_only",
            edge_top3 is not None,
            _rel(edge_path),
        ),
        _metric_row(
            "DINO_real_minus_shuffled_AUC",
            edge_real_minus_shuffled,
            "record_only",
            edge_real_minus_shuffled is not None,
            _rel(edge_path),
        ),
        _metric_row(
            "DINO_real_minus_no_temporal_AUC",
            edge_real_minus_no_temporal,
            "record_only",
            edge_real_minus_no_temporal is not None,
            _rel(edge_path),
        ),
        _metric_row(
            "best_solver_variant",
            best_s.get("variant"),
            "record_only",
            bool(best_s.get("variant")),
            _rel(solver_path),
        ),
        _metric_row("best_solver_SF50", solver_sf50, "<0.10", solver_sf50 is not None and solver_sf50 < 0.10, _rel(solver_path)),
        _metric_row("best_solver_AP50", solver_ap50, "record_only", solver_ap50 is not None, _rel(solver_path)),
        _metric_row("best_solver_GT_best_IoU", solver_gt_best, "record_only", solver_gt_best is not None, _rel(solver_path)),
        _metric_row(
            "best_solver_single_frame_rate",
            solver_single_frame,
            ">0.50",
            solver_single_frame is not None and solver_single_frame > 0.50,
            _rel(solver_path),
        ),
        _metric_row("best_solver_same_frame_violation_count", solver_violations, "=0", solver_violations == 0, _rel(solver_path)),
        _metric_row("can_enter_local2history", can_enter_local2history, "false", can_enter_local2history is False, _rel(final_path)),
    ]
    _write_csv(output_root / "fact_metric_rows.csv", metric_rows)

    required_paths = [final_path, candidate_path, edge_path, solver_path]
    missing = [_rel(path) for path in required_paths if not path.exists()]
    gate = {
        "all_inputs_exist": not missing,
        "v68_decision_NO_GO_OVERFRAGMENT": v68_decision == "NO_GO_OVERFRAGMENT",
        "candidate_bank_oracle_SF50_ge_0p50": candidate_sf50 is not None and candidate_sf50 >= 0.50,
        "DINO_edge_AUC_ge_0p80": edge_auc is not None and edge_auc >= 0.80,
        "DINO_edge_top1_precision_ge_0p85": edge_top1 is not None and edge_top1 >= 0.85,
        "best_solver_SF50_lt_0p10": solver_sf50 is not None and solver_sf50 < 0.10,
        "best_solver_single_frame_rate_gt_0p50": solver_single_frame is not None and solver_single_frame > 0.50,
        "can_enter_local2history_false": can_enter_local2history is False,
    }
    gate["pass"] = all(bool(value) for value in gate.values())
    summary = {
        "phase": "v69r2_phase0_fact_lock",
        "decision": "PASS_V68_FACT_LOCK" if gate["pass"] else "FAIL_V68_FACT_LOCK",
        "gate": gate,
        "missing_inputs": missing,
        "v68_sources": {
            "final_decision": _rel(final_path),
            "candidate_bank": _rel(candidate_path),
            "edge_audit_dinov2": _rel(edge_path),
            "local_graph_solver": _rel(solver_path),
        },
        "facts": {
            "v68_decision": v68_decision,
            "candidate_bank_oracle_SF50": candidate_sf50,
            "candidate_bank_oracle_AP50": candidate_ap50,
            "candidate_bank_GT_best_IoU": candidate_gt_best,
            "DINO_edge_AUC": edge_auc,
            "DINO_edge_top1_precision": edge_top1,
            "DINO_edge_top3_recall": edge_top3,
            "DINO_real_minus_shuffled_AUC": edge_real_minus_shuffled,
            "DINO_real_minus_no_temporal_AUC": edge_real_minus_no_temporal,
            "best_solver_variant": best_s.get("variant"),
            "best_solver_SF50": solver_sf50,
            "best_solver_AP50": solver_ap50,
            "best_solver_GT_best_IoU": solver_gt_best,
            "best_solver_single_frame_rate": solver_single_frame,
            "best_solver_same_frame_violation_count": solver_violations,
            "can_enter_local2history": can_enter_local2history,
        },
        "rows": {"fact_metric_rows_csv": _rel(output_root / "fact_metric_rows.csv")},
        "notes": [
            "Phase 0 is read-only: it imports v68 current artifacts and does not create v69r2 method predictions.",
            "The current v68 blocker is overfragment after DINO edge repair, not the earlier edge-evidence No-Go.",
        ],
    }
    _write_json(output_root / "fact_lock_summary.json", summary)
    sha_rows = []
    for path in [output_root / "fact_lock_summary.json", output_root / "fact_metric_rows.csv"]:
        sha_rows.append({"path": _rel(path), "sha256": _sha256(path), "bytes": int(path.stat().st_size)})
    _write_csv(output_root / "sha256_rows.csv", sha_rows)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="v69-r2 Phase 0: read-only v68 fact lock.")
    parser.add_argument("--output-root", default="outputs/audit/v69r2_phase0_fact_lock")
    parser.add_argument("--v68-final-decision", default="outputs/audit/v68_final_decision/final_decision.json")
    parser.add_argument("--v68-candidate-summary", default="outputs/audit/v68_candidate_bank/candidate_bank_summary.json")
    parser.add_argument("--v68-edge-summary", default="outputs/audit/v68_edge_audit_dinov2/edge_audit_summary.json")
    parser.add_argument("--v68-local-solver-summary", default="outputs/audit/v68_local_graph_solver/local_solver_summary.json")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
