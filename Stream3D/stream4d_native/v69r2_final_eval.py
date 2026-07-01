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


def _rooted(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else ROOT / path


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _anchor_row(label: str, path: Path, summary: dict[str, Any]) -> dict[str, Any]:
    best = summary.get("best_anchor_variant") or {}
    gate = summary.get("gate") or {}
    return {
        "label": label,
        "path": _rel(path),
        "decision": summary.get("decision"),
        "gate_pass": bool(gate.get("pass")),
        "best_anchor_variant": best.get("anchor_variant"),
        "anchor_oracle_SF50": best.get("anchor_oracle_SF50"),
        "anchor_oracle_AP50": best.get("anchor_oracle_AP50"),
        "anchor_GT_best_IoU_mean": best.get("anchor_GT_best_IoU_mean"),
        "anchor_D4RT_support_rate": best.get("anchor_D4RT_support_rate"),
        "anchor_underseg_rate": best.get("anchor_underseg_rate"),
        "anchor_count_per_chunk_mean": best.get("anchor_count_per_chunk_mean"),
        "anchor_count_per_frame_mean": best.get("anchor_count_per_frame_mean"),
    }


def _material_row(label: str, path: Path, summary: dict[str, Any]) -> dict[str, Any]:
    best = summary.get("best_closure_variant") or {}
    gate = summary.get("gate") or {}
    return {
        "label": label,
        "path": _rel(path),
        "decision": summary.get("decision"),
        "gate_pass": bool(gate.get("pass")),
        "candidate_source": summary.get("candidate_source", "edge_proxy"),
        "best_closure_variant": best.get("closure_variant"),
        "single_anchor_SF50": best.get("single_anchor_SF50"),
        "single_anchor_AP50": best.get("single_anchor_AP50"),
        "single_anchor_GT_best_IoU_mean": best.get("single_anchor_GT_best_IoU_mean"),
        "single_anchor_temporal_span_mean": best.get("single_anchor_temporal_span_mean"),
        "single_anchor_single_frame_rate": best.get("single_anchor_single_frame_rate"),
        "same_frame_violation_count": best.get("same_frame_cannot_link_violation_count_sum"),
        "real_minus_shuffled_SF50": summary.get("real_minus_shuffled_SF50"),
        "real_minus_no_temporal_SF50": summary.get("real_minus_no_temporal_SF50"),
        "anchor_with_visible_material_rate": best.get("anchor_with_visible_material_rate"),
        "support_mask_count_mean": best.get("support_mask_count_mean"),
        "shared_mask_count_mean": best.get("shared_mask_count_mean"),
        "underseg_bridge_rate": best.get("underseg_bridge_rate"),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_root = _rooted(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    phase0_path = _rooted(args.phase0_summary)
    phase0 = _load_json(phase0_path)
    anchor_paths = [_rooted(item.strip()) for item in str(args.anchor_summaries).split(",") if item.strip()]
    anchor_rows = [_anchor_row(f"attempt_{idx}", path, _load_json(path)) for idx, path in enumerate(anchor_paths, start=1)]
    material_paths = [_rooted(item.strip()) for item in str(args.material_summaries).split(",") if item.strip()]
    material_rows = [_material_row(f"attempt_{idx}", path, _load_json(path)) for idx, path in enumerate(material_paths, start=1)]
    _write_csv(output_root / "anchor_summary_rows.csv", anchor_rows)
    _write_csv(output_root / "material_summary_rows.csv", material_rows)

    passing = [row for row in anchor_rows if row.get("gate_pass")]
    material_passing = [row for row in material_rows if row.get("gate_pass")]
    phase0_pass = bool((phase0.get("gate") or {}).get("pass"))
    best_underseg_valid = max(
        [row for row in anchor_rows if (_float_or_none(row.get("anchor_underseg_rate")) or 999.0) <= 0.15],
        key=lambda row: float(row.get("anchor_oracle_SF50") or 0.0),
        default={},
    )
    best_oracle = max(anchor_rows, key=lambda row: float(row.get("anchor_oracle_SF50") or 0.0), default={})
    if not passing:
        decision = "NO_GO_ANCHOR_BANK"
        final_label = "NO_GO_ANCHOR_BANK"
        best = best_underseg_valid or best_oracle
        phase2_status = "not_run_blocked_by_anchor_bank"
        phase3_status = "not_run_blocked_by_anchor_bank"
    elif material_passing:
        decision = "PASS_ANCHOR_BANK"
        final_label = "CONTINUE_TO_TYPED_ASSIGNMENT"
        best = max(passing, key=lambda row: float(row.get("anchor_oracle_SF50") or 0.0))
        phase2_status = "PASS_MATERIAL_CLOSURE"
        phase3_status = "not_run_pending_typed_assignment"
    else:
        decision = "NO_GO_MATERIAL_CLOSURE"
        final_label = "NO_GO_MATERIAL_CLOSURE"
        best = max(passing, key=lambda row: float(row.get("anchor_oracle_SF50") or 0.0))
        phase2_status = "NO_GO_MATERIAL_CLOSURE"
        phase3_status = "not_run_blocked_by_material_closure"
    best_material = max(material_rows, key=lambda row: float(row.get("single_anchor_SF50") or 0.0), default={})
    summary = {
        "phase": "v69r2_final_decision",
        "decision": decision,
        "final_label": final_label,
        "phase_status": {
            "phase0_v68_fact_lock": phase0.get("decision"),
            "phase1_anchor_bank": "PASS_ANCHOR_BANK" if passing else "NO_GO_ANCHOR_BANK",
            "phase2_material_closure": phase2_status,
            "phase3_typed_assignment": phase3_status,
            "phase4_single_anchor": "not_run_blocked_by_material_closure" if decision == "NO_GO_MATERIAL_CLOSURE" else phase3_status,
            "phase5_anchor_coreference": "not_run_blocked_by_material_closure" if decision == "NO_GO_MATERIAL_CLOSURE" else phase3_status,
            "phase6_packing": "not_run_blocked_by_material_closure" if decision == "NO_GO_MATERIAL_CLOSURE" else phase3_status,
            "phase7_scene_mv_ap": "not_run_blocked_by_material_closure" if decision == "NO_GO_MATERIAL_CLOSURE" else phase3_status,
            "phase8_local2history": "not_run_blocked_by_material_closure" if decision == "NO_GO_MATERIAL_CLOSURE" else phase3_status,
            "phase9_visualization": "partial_anchor_bank_png_only_no_3d_viewer",
        },
        "gate": {
            "phase0_pass": phase0_pass,
            "phase1_anchor_bank_pass": bool(passing),
            "phase2_material_closure_pass": bool(material_passing),
            "can_enter_material_closure": bool(phase0_pass and passing),
            "can_enter_typed_assignment": bool(phase0_pass and passing and material_passing),
            "can_enter_local2history": False,
        },
        "best_anchor_attempt_any": best_oracle,
        "best_anchor_attempt_underseg_valid": best_underseg_valid,
        "selected_final_attempt": best,
        "best_material_attempt": best_material,
        "blocker_analysis": [
            "Phase 0 passed: v68 current artifact chain is DINO-edge-pass plus local overfragment.",
            "Phase 1 passed only after removing GT-derived underseg leakage from anchor objectness and using non-GT method underseg risk.",
            "Phase 2 edge-proxy material closure failed to beat v68 S11 and matched the no-temporal control.",
            "Phase 2 component-index repair exposed more anchor-centric D4RT candidates but valid non-underseg support remained too sparse after shared masks were prevented from becoming object cores.",
            "Phase 2 tracklet-index repair with D4RT guard also failed and matched its no-temporal control.",
            "Per Stop 3 and Stop 5, typed assignment, single-anchor hypothesis, packing, scene MV-AP, and local2history are blocked by material closure.",
        ],
        "repair_attempts_recorded": [
            "A1-A5 initial anchor objectness variants from high-quality, D4RT-supported, DINO-valid, repeated-signature, and area-balanced rows.",
            "A6/A7 support-balanced repair to preserve D4RT support while adding limited DINO/representative recall.",
            "A8/A9 support-floor repair to keep D4RT support >=0.50 while recovering oracle recall.",
            "A10 final underseg cap repair to enforce underseg <=0.15 after support-floor and per-frame cap.",
            "Repair5 replaced GT-derived underseg risk with method-only material/DINO/overlap risk and produced a passing non-GT anchor bank.",
            "Phase2 repair1 replaced sparse v68 edge-row candidate reuse with D4RT component-index closure.",
            "Phase2 repair2 made underseg/shared support non-bridging and added relaxed component recall controls.",
            "Phase2 repair3 tried DINO-mode/repeated-signature tracklet closure with D4RT guard and matched controls.",
        ],
        "source_summaries": {
            "phase0": _rel(phase0_path),
            "anchor_attempts": [_rel(path) for path in anchor_paths],
            "material_attempts": [_rel(path) for path in material_paths],
        },
        "rows": {
            "anchor_summary_rows_csv": _rel(output_root / "anchor_summary_rows.csv"),
            "material_summary_rows_csv": _rel(output_root / "material_summary_rows.csv"),
        },
        "notes": [
            "This final decision stops at Phase 2 by plan rule; it does not claim typed support, packing, scene MV-AP, or local2history results.",
            "Anchor oracle metrics are diagnostic-only and forbidden for method tables because they use GT majority mapping for evaluation.",
        ],
    }
    _write_json(output_root / "final_decision.json", summary)
    sha_rows = []
    for path in [output_root / "final_decision.json", output_root / "anchor_summary_rows.csv", output_root / "material_summary_rows.csv"]:
        sha_rows.append({"path": _rel(path), "sha256": _sha256(path), "bytes": int(path.stat().st_size)})
    _write_csv(output_root / "sha256_rows.csv", sha_rows)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="v69-r2 final decision builder.")
    parser.add_argument("--output-root", default="outputs/audit/v69r2_final_decision")
    parser.add_argument("--phase0-summary", default="outputs/audit/v69r2_phase0_fact_lock/fact_lock_summary.json")
    parser.add_argument(
        "--anchor-summaries",
        default=",".join(
            [
                "outputs/audit/v69r2_anchor_bank/anchor_bank_summary.json",
                "outputs/audit/v69r2_anchor_bank_repair1/anchor_bank_summary.json",
                "outputs/audit/v69r2_anchor_bank_repair2/anchor_bank_summary.json",
                "outputs/audit/v69r2_anchor_bank_repair3/anchor_bank_summary.json",
                "outputs/audit/v69r2_anchor_bank_repair4/anchor_bank_summary.json",
                "outputs/audit/v69r2_anchor_bank_repair5_nogt_underseg/anchor_bank_summary.json",
            ]
        ),
    )
    parser.add_argument(
        "--material-summaries",
        default=",".join(
            [
                "outputs/audit/v69r2_material_closure/closure_summary.json",
                "outputs/audit/v69r2_material_closure_repair2_shared_no_bridge_probe5/closure_summary.json",
                "outputs/audit/v69r2_material_closure_repair3_tracklet_probe5/closure_summary.json",
            ]
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
