from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.run_v65_scene_multiview_ap import _sha256, _write_csv, _write_json  # noqa: E402
from tools.run_v66_scene_mv_ap_probe5 import _rel  # noqa: E402


def _project(path: str | Path) -> Path:
    path_obj = Path(path)
    if path_obj.is_absolute():
        return path_obj
    if path_obj.parts and path_obj.parts[0] == ROOT.name:
        return REPO_ROOT / path_obj
    return ROOT / path_obj


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(_project(path).read_text(encoding="utf-8"))


def _float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except Exception:
        return None


def _bool(value: Any) -> bool:
    return bool(value)


def _row(name: str, value: Any, pass_value: bool | None, source: str) -> dict[str, Any]:
    return {"metric": name, "value": value, "pass": pass_value, "source": source}


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_root = _project(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    paths = {
        "final": "outputs/audit/v67_final_decision/final_decision.json",
        "phase0": "outputs/audit/v67_phase0_metric_relock/metric_relock_summary.json",
        "phase1": "outputs/audit/v67_mask_universe/mask_universe_summary.json",
        "phase2": "outputs/audit/v67_ledger_join/ledger_join_summary.json",
        "phase3": "outputs/audit/v67_local_baselines/local_baseline_summary.json",
        "phase4": "outputs/audit/v67_object_balanced_setcover/setcover_summary.json",
        "phase5": "outputs/audit/v67_local_mask_graph/local_graph_summary.json",
        "casebook": "outputs/audit/v67_visual_casebook/casebook_summary.json",
    }
    missing = [{"name": name, "path": path} for name, path in paths.items() if not _project(path).exists()]
    if missing:
        _write_csv(output_root / "missing_input_rows.csv", missing)
        summary = {
            "phase": "v68_phase0_fact_lock",
            "decision": "FAIL_MISSING_V67_INPUTS",
            "gate": {"pass": False, "all_inputs_present": False},
            "missing_inputs": missing,
        }
        _write_json(output_root / "fact_lock_summary.json", summary)
        print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
        return summary

    final = _read_json(paths["final"])
    phase0 = _read_json(paths["phase0"])
    phase1 = _read_json(paths["phase1"])
    phase2 = _read_json(paths["phase2"])
    phase3 = _read_json(paths["phase3"])
    phase4 = _read_json(paths["phase4"])
    phase5 = _read_json(paths["phase5"])
    casebook = _read_json(paths["casebook"])
    final_metrics = final.get("key_metrics") or {}
    final_gates = final.get("hard_gates") or {}

    v67_decision = final.get("decision")
    can_enter_local2history = bool(final.get("can_enter_local2history"))
    metric_relock_pass = bool(final_gates.get("metric_relock_pass") or (phase0.get("gate") or {}).get("pass"))
    ledger_join_pass = bool(final_gates.get("ledger_join_pass") or (phase2.get("gate") or {}).get("pass"))
    u0_sf50 = _float(phase1.get("u0_selected_oracle_sf50_mean"))
    u2_sf50 = _float(phase1.get("u2_representative_oracle_sf50_mean"))
    u3_sf50 = _float(phase1.get("u3_raw_cropformer_oracle_sf50_mean"))
    phase3_selected_sf50 = _float((phase3.get("oracle_headroom") or {}).get("b8_selected_oracle_sf50"))
    phase3_raw_sf50 = _float((phase3.get("oracle_headroom") or {}).get("b7_raw_oracle_sf50"))
    phase4_best_k_sf50 = _float((phase4.get("best_K") or {}).get("local_score_free_match50_recall_mean"))
    phase5_best_g_sf50 = _float((phase5.get("best_G") or {}).get("local_score_free_match50_recall_mean"))
    phase5_best_g_ap50 = _float((phase5.get("best_G") or {}).get("local_AP50_mean"))
    phase5_best_g_single = _float((phase5.get("best_G") or {}).get("single_frame_object_rate_mean"))
    viewer_scene_count = int(casebook.get("viewer_scene_count") or 0)

    gate = {
        "all_inputs_present": True,
        "v67_decision_is_no_go_local_graph_solver": v67_decision == "NO_GO_LOCAL_GRAPH_SOLVER_BLOCKER",
        "can_enter_local2history_false": can_enter_local2history is False,
        "metric_relock_pass": metric_relock_pass,
        "ledger_join_pass": ledger_join_pass,
        "u2_representative_oracle_sf50_ge_0p50": u2_sf50 is not None and u2_sf50 >= 0.50,
        "phase5_best_G_SF50_lt_0p30": phase5_best_g_sf50 is not None and phase5_best_g_sf50 < 0.30,
    }
    gate["pass"] = all(bool(value) for value in gate.values())
    rows = [
        _row("v67_decision", v67_decision, gate["v67_decision_is_no_go_local_graph_solver"], paths["final"]),
        _row("can_enter_local2history", can_enter_local2history, gate["can_enter_local2history_false"], paths["final"]),
        _row("metric_relock_pass", metric_relock_pass, gate["metric_relock_pass"], paths["phase0"]),
        _row("ledger_join_pass", ledger_join_pass, gate["ledger_join_pass"], paths["phase2"]),
        _row("phase1_decision", phase1.get("decision"), phase1.get("decision") == "CANDIDATE_UNIVERSE_LOSS", paths["phase1"]),
        _row("phase3_decision", phase3.get("decision"), phase3.get("decision") == "ORACLE_HEADROOM_HIGH_NON_ORACLE_LOCAL_GATE_FAIL", paths["phase3"]),
        _row("phase4_decision", phase4.get("decision"), phase4.get("decision") == "OBJECT_BALANCED_SETCOVER_FAILS_LOCAL_GATE", paths["phase4"]),
        _row("phase5_decision", phase5.get("decision"), phase5.get("decision") == "LOCAL_MASK_GRAPH_FAILS_LOCAL_GATE", paths["phase5"]),
        _row("U0_current_selected_oracle_SF50", u0_sf50, None, paths["phase1"]),
        _row("U2_representative_oracle_SF50", u2_sf50, gate["u2_representative_oracle_sf50_ge_0p50"], paths["phase1"]),
        _row("U3_raw_oracle_SF50", u3_sf50, u3_sf50 is not None and u3_sf50 >= 0.50, paths["phase1"]),
        _row("phase3_selected_oracle_SF50", phase3_selected_sf50, None, paths["phase3"]),
        _row("phase3_raw_oracle_SF50", phase3_raw_sf50, phase3_raw_sf50 is not None and phase3_raw_sf50 >= 0.50, paths["phase3"]),
        _row("phase4_best_K_SF50", phase4_best_k_sf50, None, paths["phase4"]),
        _row("phase5_best_G_SF50", phase5_best_g_sf50, gate["phase5_best_G_SF50_lt_0p30"], paths["phase5"]),
        _row("phase5_best_G_AP50", phase5_best_g_ap50, None, paths["phase5"]),
        _row("phase5_best_G_single_frame_object_rate", phase5_best_g_single, None, paths["phase5"]),
        _row("viewer_scene_count", viewer_scene_count, viewer_scene_count == 0, paths["casebook"]),
    ]
    _write_csv(output_root / "v67_metric_rows.csv", rows)
    _write_csv(output_root / "missing_input_rows.csv", [])
    summary = {
        "phase": "v68_phase0_fact_lock",
        "decision": "PASS_V67_FACT_LOCK" if gate["pass"] else "FAIL_V67_FACT_LOCK",
        "diagnostic_only": True,
        "gate": gate,
        "key_metrics": {row["metric"]: row["value"] for row in rows},
        "v67_final_key_metrics": final_metrics,
        "inputs": paths,
        "rows": {
            "v67_metric_rows_csv": _rel(output_root / "v67_metric_rows.csv"),
            "missing_input_rows_csv": _rel(output_root / "missing_input_rows.csv"),
        },
    }
    _write_json(output_root / "fact_lock_summary.json", summary)
    sha_rows = []
    for path in [output_root / "fact_lock_summary.json", output_root / "v67_metric_rows.csv", output_root / "missing_input_rows.csv"]:
        sha_rows.append({"path": _rel(path), "sha256": _sha256(path), "bytes": int(path.stat().st_size)})
    _write_csv(output_root / "sha256_rows.csv", sha_rows)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run v68 Phase 0 fact lock from v67 summaries.")
    parser.add_argument("--output-root", default="outputs/audit/v68_phase0_fact_lock")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
