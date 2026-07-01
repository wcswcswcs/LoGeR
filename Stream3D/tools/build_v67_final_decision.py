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
from tools.run_v66_scene_mv_ap_probe5 import _rel  # noqa: E402


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_root = ROOT / args.output_root if not Path(args.output_root).is_absolute() else Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    phase0 = _read_json(ROOT / "outputs/audit/v67_phase0_metric_relock/metric_relock_summary.json")
    phase1 = _read_json(ROOT / "outputs/audit/v67_mask_universe/mask_universe_summary.json")
    phase2 = _read_json(ROOT / "outputs/audit/v67_ledger_join/ledger_join_summary.json")
    phase3 = _read_json(ROOT / "outputs/audit/v67_local_baselines/local_baseline_summary.json")
    phase4 = _read_json(ROOT / "outputs/audit/v67_object_balanced_setcover/setcover_summary.json")
    phase5 = _read_json(ROOT / "outputs/audit/v67_local_mask_graph/local_graph_summary.json")
    casebook = _read_json(ROOT / "outputs/audit/v67_visual_casebook/casebook_summary.json")
    final_decision = {
        "phase": "v67_final_decision",
        "decision": "NO_GO_LOCAL_GRAPH_SOLVER_BLOCKER",
        "can_enter_local2history": False,
        "no_go_reason": "Local graph solver failed local gate while raw/representative oracle headroom remains high.",
        "hard_gates": {
            "metric_relock_pass": bool((phase0.get("gate") or {}).get("pass")),
            "ledger_join_pass": bool((phase2.get("gate") or {}).get("pass")),
            "phase3_best_non_oracle_local_gate_pass": bool((phase3.get("gate") or {}).get("best_non_oracle_local_gate_pass")),
            "phase4_best_K_local_gate_pass": bool((phase4.get("gate") or {}).get("best_K_local_gate_pass")),
            "phase5_best_G_local_gate_pass": bool((phase5.get("gate") or {}).get("best_G_local_gate_pass")),
            "phase7_3d_viewer_gate_pass": bool((casebook.get("gate") or {}).get("viewer_scene_count_ge_5")),
        },
        "key_metrics": {
            "phase1_decision": phase1.get("decision"),
            "phase3_decision": phase3.get("decision"),
            "phase4_decision": phase4.get("decision"),
            "phase5_decision": phase5.get("decision"),
            "U2_representative_oracle_SF50": phase1.get("u2_representative_oracle_sf50_mean"),
            "phase3_raw_oracle_SF50": (phase3.get("oracle_headroom") or {}).get("b7_raw_oracle_sf50"),
            "phase3_selected_oracle_SF50": (phase3.get("oracle_headroom") or {}).get("b8_selected_oracle_sf50"),
            "phase3_raw_minus_selected_SF50": (phase3.get("oracle_headroom") or {}).get("b7_raw_minus_b8_selected_sf50"),
            "phase4_best_K": (phase4.get("best_K") or {}).get("variant"),
            "phase4_best_K_SF50": (phase4.get("best_K") or {}).get("local_score_free_match50_recall_mean"),
            "phase5_best_G": (phase5.get("best_G") or {}).get("variant"),
            "phase5_best_G_SF50": (phase5.get("best_G") or {}).get("local_score_free_match50_recall_mean"),
            "phase5_best_G_AP50": (phase5.get("best_G") or {}).get("local_AP50_mean"),
            "phase5_best_G_GT_best_IoU": (phase5.get("best_G") or {}).get("local_GT_best_IoU_mean_mean"),
            "phase5_best_G_single_frame_object_rate": (phase5.get("best_G") or {}).get("single_frame_object_rate_mean"),
            "phase5_G7_oracle_SF50": 0.5245689920285618,
            "casebook_case_count": casebook.get("case_count"),
            "viewer_scene_count": casebook.get("viewer_scene_count"),
        },
        "evidence_chain": [
            "Phase 0 passed metric relock and no-GT prediction-path audit.",
            "Phase 1 showed raw/representative masks have high oracle headroom and selected/reprojection candidates lose it.",
            "Phase 2 ruled out ledger join key/stale-path/missing-mask engineering blockers for ledger-backed candidates.",
            "Phase 3 showed chunk-local raw/representative oracle headroom remains high while non-oracle baselines fail local gate.",
            "Phase 4 showed object-balanced set-cover K0-K7 does not pass local gate.",
            "Phase 5 showed local graph G0-G6 does not pass local gate; best G2 improves SF50 but overfragments heavily.",
            "Phase 7 produced a 2D casebook but did not satisfy the requested 3D viewer gate.",
        ],
        "next_repair_direction": [
            "Do not evaluate local2history as method success before local gate passes.",
            "Add stronger non-GT cross-frame association cues: semantic mode atoms, material residual consistency, repeated signature clustering, and calibrated D4RT support edges.",
            "Build the missing 3D Viser viewer before claiming full visual confirmation.",
        ],
        "outputs": {
            "phase0": "outputs/audit/v67_phase0_metric_relock/metric_relock_summary.json",
            "phase1": "outputs/audit/v67_mask_universe/mask_universe_summary.json",
            "phase2": "outputs/audit/v67_ledger_join/ledger_join_summary.json",
            "phase3": "outputs/audit/v67_local_baselines/local_baseline_summary.json",
            "phase4": "outputs/audit/v67_object_balanced_setcover/setcover_summary.json",
            "phase5": "outputs/audit/v67_local_mask_graph/local_graph_summary.json",
            "phase7": "outputs/audit/v67_visual_casebook/casebook_summary.json",
        },
    }
    _write_json(output_root / "final_decision.json", final_decision)
    rows = [{"field": key, "value": value} for key, value in final_decision["key_metrics"].items()]
    _write_csv(output_root / "final_decision_key_metrics.csv", rows)
    sha_rows = []
    for path in [output_root / "final_decision.json", output_root / "final_decision_key_metrics.csv"]:
        sha_rows.append({"path": _rel(path), "sha256": _sha256(path), "bytes": int(path.stat().st_size)})
    _write_csv(output_root / "sha256_rows.csv", sha_rows)
    print(json.dumps(final_decision, indent=2, sort_keys=True), flush=True)
    return final_decision


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build v67 final decision artifact.")
    parser.add_argument("--output-root", default="outputs/audit/v67_final_decision")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
