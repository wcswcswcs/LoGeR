from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
AUDIT_ROOT = ROOT / "Stream3D" / "outputs" / "audit"
OUT_DIR = AUDIT_ROOT / "v101_final_closeout"

PHASES = {
    "phase0": AUDIT_ROOT / "v101_phase0_fact_lock" / "summary.json",
    "phase1": AUDIT_ROOT / "v101_phase1_f2_fragmentation_casebook" / "summary.json",
    "phase1b": AUDIT_ROOT / "v101_phase1b_fragment_quality_decomp" / "summary.json",
    "phase2": AUDIT_ROOT / "v101_phase2_geometry_provider_capability" / "summary.json",
    "phase2b": AUDIT_ROOT / "v101_phase2b_false_bridge_repair_probe" / "summary.json",
}


def _rel(path: Path | str) -> str:
    p = Path(path)
    try:
        return str(p.resolve().relative_to(ROOT))
    except Exception:
        return str(p)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return _rel(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    summaries = {name: _read_json(path) for name, path in PHASES.items()}
    phase0 = summaries["phase0"]
    phase1 = summaries["phase1"]
    phase1b = summaries["phase1b"]
    phase2 = summaries["phase2"]
    phase2b = summaries["phase2b"]

    provider_bridge_ok = bool(phase2.get("provider_bridge_potential_confirmed")) or bool(
        phase2b.get("provider_bridge_potential_confirmed_after_repair")
    )
    phase3_allowed = bool(phase1.get("fragmentation_confirmed")) and provider_bridge_ok

    final_decision = (
        "PASS_ENTER_PHASE3_FRAGMENT_REPAIR"
        if phase3_allowed
        else "FINAL_NO_GO_PHASE3_BLOCKED_PROVIDER_BRIDGE_FALSE_BRIDGE_AND_PURITY"
    )

    payload = {
        "schema_version": "stream4d_v101_final_closeout_summary_v1",
        "phase_id": "v101_final_closeout",
        "created_unix_time": time.time(),
        "plan_execution_completed": True,
        "method_success_claim_allowed": False,
        "phase3_fragment_repair_allowed": phase3_allowed,
        "phase3_fragment_repair_run": False,
        "final_decision": final_decision,
        "truthfulness_note": "No Phase3 AP numbers are reported because the plan gate did not allow Phase3 fragment repair.",
        "phase_decisions": {
            "phase0": phase0.get("decision"),
            "phase1": phase1.get("decision"),
            "phase1b": phase1b.get("decision"),
            "phase2": phase2.get("decision"),
            "phase2b": phase2b.get("decision"),
        },
        "key_metrics": {
            "phase0_f2_holdout_MV_AP_window": phase0.get("F2_phase2c_holdout_MV_AP_window"),
            "phase0_f2_holdout_MV_AP_scene_fragmented": phase0.get("F2_phase2c_holdout_MV_AP_scene_fragmented"),
            "phase1_GT_fragment_count_ge2_rate": phase1.get("GT_fragment_count_ge2_rate"),
            "phase1_union_minus_best_IoU_mean": phase1.get("union_minus_best_IoU_mean"),
            "phase1b_effective_iou0p05_fragment_ge2_rate": phase1b.get("effective_iou0p05_fragment_ge2_rate"),
            "phase2_d4rt_recall_at_0p10": phase2.get("G0_D4RT_bridge_at_0p10", {}).get(
                "same_object_bridge_recall_at_tau"
            ),
            "phase2_d4rt_false_bridge_at_0p10": phase2.get("G0_D4RT_bridge_at_0p10", {}).get(
                "false_bridge_rate_same_semantic_diff_GT_at_tau"
            ),
            "phase2_d4rt_bridge_AUC": phase2.get("G0_D4RT_bridge_at_0p10", {}).get("bridge_AUC_diagnostic"),
            "phase2b_best_tau0p10_filter": phase2b.get("best_tau0p10_row", {}).get("filter_name"),
            "phase2b_best_tau0p10_recall": phase2b.get("best_tau0p10_row", {}).get(
                "same_object_bridge_recall_at_tau"
            ),
            "phase2b_best_tau0p10_false_bridge": phase2b.get("best_tau0p10_row", {}).get(
                "false_bridge_rate_at_tau"
            ),
            "phase2b_passing_filter_count": phase2b.get("passing_filter_count"),
        },
        "blocked_repair_directions": phase2b.get("artifact_schema_blockers", []),
        "analysis": {
            "main_conclusion": (
                "F2 local/window baseline remains the only allowed positive claim. "
                "Fragment repair was not run because no audited provider/filter satisfied bridge recall, false bridge, and purity requirements."
            ),
            "evidence_chain": [
                "Phase0 locked v100 Phase2c F2 as canonical baseline without evaluator drift.",
                "Phase1 confirmed raw fragmentation but Phase1b showed most fragments are tiny/low-quality overlaps and direct union merge is harmful.",
                "Phase2 found DA3-BASE mask support coverage but no persistent bridge primitive/purity rows; D4RT had bridge recall/AUC but high false bridge and no surfel purity.",
                "Phase2b tried plan-directed false-bridge filters; none reached the v101 bridge gate.",
            ],
        },
        "source_summaries": {name: _rel(path) for name, path in PHASES.items()},
        "code_changes": [
            "Stream3D/tools/build_v101_phase0_fact_lock.py",
            "Stream3D/tools/build_v101_phase1_f2_fragmentation_casebook.py",
            "Stream3D/tools/build_v101_phase1b_fragment_quality_decomp.py",
            "Stream3D/tools/build_v101_phase2_geometry_provider_capability.py",
            "Stream3D/tools/build_v101_phase2b_false_bridge_repair_probe.py",
            "Stream3D/tools/build_v101_final_closeout.py",
            "docs/stream4d_v101_执行日志.md",
            "docs/stream4d_v101_实验结果复盘.md",
        ],
        "summary": _rel(OUT_DIR / "summary.json"),
    }
    _write_json(OUT_DIR / "summary.json", payload)
    print(json.dumps(_jsonable(payload), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
