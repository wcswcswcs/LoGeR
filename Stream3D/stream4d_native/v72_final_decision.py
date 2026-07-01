from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _rooted(path_text: str | Path) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else ROOT / path


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or str(value).strip() == "":
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_root = _rooted(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    paths = {
        "phase0_fact_lock": _rooted(args.phase0_summary),
        "phase1_signal_adequacy": _rooted(args.phase1_summary),
        "phase2_semantic_proposals": _rooted(args.phase2_summary),
        "phase2_broad_casebook": _rooted(args.phase2_casebook),
        "phase2_dense_token_no_area_floor": _rooted(args.phase2_dense_no_area_floor_summary),
        "phase2_dense_token_area_bin": _rooted(args.phase2_dense_area_bin_summary),
        "phase2_sam2_source_filtered": _rooted(args.phase2_sam2_source_filtered_summary),
        "phase2_sam2_source_relaxed": _rooted(args.phase2_sam2_source_relaxed_summary),
        "phase3_d4rt_no_area_floor": _rooted(args.phase3_d4rt_no_area_floor_summary),
        "phase3_d4rt_area_bin": _rooted(args.phase3_d4rt_area_bin_summary),
        "phase4_objectness_no_area_floor": _rooted(args.phase4_objectness_no_area_floor_summary),
        "phase4_objectness_area_bin": _rooted(args.phase4_objectness_area_bin_summary),
        "phase5_setcover_no_area_floor": _rooted(args.phase5_setcover_no_area_floor_summary),
        "phase5_setcover_area_bin": _rooted(args.phase5_setcover_area_bin_summary),
    }
    data = {name: _load_json(path) for name, path in paths.items()}
    missing = [{"phase": name, "path": _rel(path)} for name, path in paths.items() if not path.exists()]
    phase0 = data["phase0_fact_lock"]
    phase1 = data["phase1_signal_adequacy"]
    phase2 = data["phase2_semantic_proposals"]
    casebook = data["phase2_broad_casebook"]
    dense_no_area = data["phase2_dense_token_no_area_floor"]
    dense_area_bin = data["phase2_dense_token_area_bin"]
    sam2_source_filtered = data["phase2_sam2_source_filtered"]
    sam2_source_relaxed = data["phase2_sam2_source_relaxed"]
    d4rt_no_area = data["phase3_d4rt_no_area_floor"]
    d4rt_area_bin = data["phase3_d4rt_area_bin"]
    objectness_no_area = data["phase4_objectness_no_area_floor"]
    objectness_area_bin = data["phase4_objectness_area_bin"]
    setcover_no_area = data["phase5_setcover_no_area_floor"]
    setcover_area_bin = data["phase5_setcover_area_bin"]

    method_uses_gt = bool((phase0.get("key_metrics") or {}).get("method_prediction_uses_gt_anywhere"))
    phase2_pass = bool((phase2.get("gate") or {}).get("pass"))
    dense_no_area_pass = bool((dense_no_area.get("gate") or {}).get("pass"))
    dense_area_bin_pass = bool((dense_area_bin.get("gate") or {}).get("pass"))
    dense_any_pass = dense_no_area_pass or dense_area_bin_pass
    phase3_any_pass = bool((d4rt_no_area.get("gate") or {}).get("pass")) or bool((d4rt_area_bin.get("gate") or {}).get("pass"))
    phase4_any_pass = bool((objectness_no_area.get("gate") or {}).get("pass")) or bool((objectness_area_bin.get("gate") or {}).get("pass"))
    phase5_no_area_pass = bool((setcover_no_area.get("gate") or {}).get("pass"))
    phase5_area_bin_pass = bool((setcover_area_bin.get("gate") or {}).get("pass"))
    phase5_any_pass = phase5_no_area_pass or phase5_area_bin_pass
    phase1_semantic = bool((phase1.get("gate") or {}).get("semantic_pass"))
    if method_uses_gt:
        primary_blocker = "METHOD_BOUNDARY_VIOLATION"
        final_label = "NO_GO_METHOD_BOUNDARY_VIOLATION"
    elif phase4_any_pass and not phase5_any_pass:
        primary_blocker = "PHASE5_PROPOSAL_SETCOVER_TEMPORAL_GROUP_RISK_COVERAGE_FAILURE"
        final_label = "NO_GO_PHASE5_PROPOSAL_SETCOVER"
    elif phase1_semantic and not (phase2_pass or dense_any_pass):
        primary_blocker = "PROPOSAL_GENERATOR_INSUFFICIENT"
        final_label = "NO_GO_PROPOSAL_GENERATOR_INSUFFICIENT"
    else:
        primary_blocker = "INCONCLUSIVE_NEEDS_TARGETED_DIAGNOSTIC"
        final_label = "NO_GO_INCONCLUSIVE"

    def _best_setcover_summary() -> dict[str, Any]:
        candidates = [setcover_no_area.get("best_method") or {}, setcover_area_bin.get("best_method") or {}]
        return max(candidates, key=lambda row: _float(row.get("representative_proposal_oracle_SF50_diagnostic"), -1.0), default={})

    best_setcover = _best_setcover_summary()

    phase_rows = [
        {
            "phase": "phase0_fact_lock",
            "decision": phase0.get("decision"),
            "status": "pass" if bool(phase0.get("can_enter_v72_phase1")) else "fail",
            "evidence_path": _rel(paths["phase0_fact_lock"]),
        },
        {
            "phase": "phase1_signal_adequacy",
            "decision": phase1.get("decision"),
            "status": "partial_go" if phase1_semantic else "fail",
            "evidence_path": _rel(paths["phase1_signal_adequacy"]),
        },
        {
            "phase": "phase2_semantic_proposals",
            "decision": phase2.get("decision"),
            "status": "fail",
            "evidence_path": _rel(paths["phase2_semantic_proposals"]),
        },
        {
            "phase": "phase2_dense_token_no_area_floor",
            "decision": dense_no_area.get("decision"),
            "status": "pass" if dense_no_area_pass else "fail",
            "evidence_path": _rel(paths["phase2_dense_token_no_area_floor"]),
        },
        {
            "phase": "phase2_dense_token_area_bin",
            "decision": dense_area_bin.get("decision"),
            "status": "pass" if dense_area_bin_pass else "fail",
            "evidence_path": _rel(paths["phase2_dense_token_area_bin"]),
        },
        {
            "phase": "phase2_sam2_source_filtered",
            "decision": sam2_source_filtered.get("decision"),
            "status": "diagnostic_partial" if bool(sam2_source_filtered.get("source_diagnostic_supports_stronger_mask_source")) else "diagnostic_no_go",
            "evidence_path": _rel(paths["phase2_sam2_source_filtered"]),
        },
        {
            "phase": "phase2_sam2_source_relaxed",
            "decision": sam2_source_relaxed.get("decision"),
            "status": "diagnostic_partial" if bool(sam2_source_relaxed.get("source_diagnostic_supports_stronger_mask_source")) else "diagnostic_no_go",
            "evidence_path": _rel(paths["phase2_sam2_source_relaxed"]),
        },
        {
            "phase": "phase3_d4rt_verification",
            "decision": f"{d4rt_no_area.get('decision')} / {d4rt_area_bin.get('decision')}",
            "status": "pass" if phase3_any_pass else "diagnostic_no_go",
            "evidence_path": f"{_rel(paths['phase3_d4rt_no_area_floor'])};{_rel(paths['phase3_d4rt_area_bin'])}",
        },
        {
            "phase": "phase4_objectness_ranking",
            "decision": f"{objectness_no_area.get('decision')} / {objectness_area_bin.get('decision')}",
            "status": "pass" if phase4_any_pass else "fail",
            "evidence_path": f"{_rel(paths['phase4_objectness_no_area_floor'])};{_rel(paths['phase4_objectness_area_bin'])}",
        },
        {
            "phase": "phase5_proposal_setcover",
            "decision": f"{setcover_no_area.get('decision')} / {setcover_area_bin.get('decision')}",
            "status": "pass" if phase5_any_pass else "fail",
            "evidence_path": f"{_rel(paths['phase5_setcover_no_area_floor'])};{_rel(paths['phase5_setcover_area_bin'])}",
        },
        {
            "phase": "phase6_local_birth",
            "decision": "not_run_blocked_by_phase5_proposal_setcover",
            "status": "not_run",
            "evidence_path": "",
        },
        {
            "phase": "phase8_scene_mv_ap",
            "decision": "not_run_blocked_by_local_gate",
            "status": "not_run",
            "evidence_path": "",
        },
        {
            "phase": "phase9_visual_casebook",
            "decision": casebook.get("decision"),
            "status": "partial_phase2_casebook_only",
            "evidence_path": _rel(paths["phase2_broad_casebook"]),
        },
    ]
    decision_matrix_rows = [
        {
            "case": "A",
            "condition": "semantic signal present but proposal generation fails to pass after mask-level and dense-token repairs",
            "matched": primary_blocker == "PROPOSAL_GENERATOR_INSUFFICIENT",
            "conclusion": "semantic signal exists but current proposal generator/objectness usage is insufficient",
            "evidence": (
                f"mask_level_gain={phase2.get('best_minus_SP0_oracle_SF50')}; "
                f"dense_no_area_gain={dense_no_area.get('best_dense_minus_SP0_oracle_SF50')}; "
                f"dense_area_bin_gain={dense_area_bin.get('best_dense_minus_SP0_oracle_SF50')}"
            ),
        },
        {
            "case": "D/F",
            "condition": "D4RT/fusion contribution not proven at candidate-ranking stage",
            "matched": not bool((phase1.get("gate") or {}).get("fusion_pass")),
            "conclusion": "D4RT must remain verifier/control candidate until Phase3 real-vs-shuffled evidence exists",
            "evidence": f"fusion_minus_best_single_top64={((phase1.get('key_metrics') or {}).get('fusion_minus_best_single_top64_iou50_rate'))}",
        },
        {
            "case": "I",
            "condition": "proposal-level oracle/object quality gates do not both exceed existing mask baseline",
            "matched": not (phase2_pass or dense_any_pass),
            "conclusion": "proposal universe remains insufficient after dense token repair; stronger segmentation/proposal source is needed",
            "evidence": (
                f"no_area_floor: SF50_gain={dense_no_area.get('best_dense_minus_SP0_oracle_SF50')} "
                f"majority_gain={dense_no_area.get('best_dense_minus_SP0_majority_IoU')}; "
                f"area_bin: SF50_gain={dense_area_bin.get('best_dense_minus_SP0_oracle_SF50')} "
                f"majority_gain={dense_area_bin.get('best_dense_minus_SP0_majority_IoU')}"
            ),
        },
        {
            "case": "Source",
            "condition": "plan-directed stronger segmentation source check after dense broad decomposition failure",
            "matched": True,
            "conclusion": "available v51 SAM2 4-frame stacks do not outperform CropFormer on targeted source-ceiling diagnostics; direct SAM2 replacement is not a valid repair from current artifacts",
            "evidence": (
                f"filtered_sam2_minus_crop_gt_best={sam2_source_filtered.get('sam2_minus_cropformer_gt_best_IoU_mean')}; "
                f"filtered_sam2_minus_crop_oracle_sf50={sam2_source_filtered.get('sam2_minus_cropformer_source_oracle_SF50')}; "
                f"relaxed_sam2_minus_crop_gt_best={sam2_source_relaxed.get('sam2_minus_cropformer_gt_best_IoU_mean')}; "
                f"relaxed_sam2_minus_crop_oracle_sf50={sam2_source_relaxed.get('sam2_minus_cropformer_source_oracle_SF50')}"
            ),
        },
        {
            "case": "Phase5",
            "condition": "proposal-level set cover and temporal group repair fail Phase5 pass gates",
            "matched": phase4_any_pass and not phase5_any_pass,
            "conclusion": "Phase4 objectness is usable for ranking, but proposal-level set cover cannot satisfy object quality, coverage, D4RT coverage, and broad/underseg risk together",
            "evidence": (
                f"best_variant={best_setcover.get('variant')}; "
                f"SF50={best_setcover.get('representative_proposal_oracle_SF50_diagnostic')}; "
                f"GT_best_IoU={best_setcover.get('representative_proposal_GT_best_IoU_mean_diagnostic')}; "
                f"coverage={best_setcover.get('covered_total_atom_weight_ratio')}; "
                f"D4RT_coverage={best_setcover.get('covered_D4RT_atom_weight_ratio')}; "
                f"risk={best_setcover.get('unresolved_broad_underseg_rate')}"
            ),
        },
    ]
    summary = {
        "phase": "v72_final_decision",
        "decision": final_label,
        "primary_blocker": primary_blocker,
        "secondary_blocker": "D4RT_FUSION_NOT_PROVEN",
        "phase0_fact_lock": phase0.get("decision"),
        "phase1_signal_adequacy": phase1.get("decision"),
        "phase2_semantic_proposals": phase2.get("decision"),
        "phase2_dense_token_no_area_floor": dense_no_area.get("decision"),
        "phase2_dense_token_area_bin": dense_area_bin.get("decision"),
        "phase2_sam2_source_filtered": sam2_source_filtered.get("decision"),
        "phase2_sam2_source_relaxed": sam2_source_relaxed.get("decision"),
        "phase3_d4rt_verification": f"{d4rt_no_area.get('decision')} / {d4rt_area_bin.get('decision')}",
        "phase4_objectness_ranking": f"{objectness_no_area.get('decision')} / {objectness_area_bin.get('decision')}",
        "phase5_proposal_setcover": f"{setcover_no_area.get('decision')} / {setcover_area_bin.get('decision')}",
        "phase6_local_birth": "not_run_blocked_by_phase5_proposal_setcover",
        "phase7_controls": "early_decision_from_phase1_phase2_controls",
        "phase8_scene_mv_ap": "not_run_blocked_by_local_gate",
        "phase9_visual_casebook": "partial_phase2_broad_casebook_done",
        "can_enter_local2history": False,
        "method_uses_gt_anywhere": method_uses_gt,
        "best_method_local_SF50": None,
        "best_method_AP50": None,
        "best_method_GT_best_IoU": None,
        "fusion_minus_semantic": (phase1.get("key_metrics") or {}).get("fusion_minus_best_single_top64_iou50_rate"),
        "fusion_minus_D4RT": None,
        "real_minus_shuffled_D4RT": None,
        "real_minus_no_temporal_D4RT": None,
        "phase2_best_minus_SP0_oracle_SF50": phase2.get("best_minus_SP0_oracle_SF50"),
        "phase2_best_minus_SP0_GT_best_IoU": phase2.get("best_minus_SP0_GT_best_IoU"),
        "phase2_dense_no_area_best_variant": dense_no_area.get("best_dense_variant"),
        "phase2_dense_no_area_best_minus_SP0_oracle_SF50": dense_no_area.get("best_dense_minus_SP0_oracle_SF50"),
        "phase2_dense_no_area_best_minus_SP0_majority_IoU": dense_no_area.get("best_dense_minus_SP0_majority_IoU"),
        "phase2_dense_area_bin_best_variant": dense_area_bin.get("best_dense_variant"),
        "phase2_dense_area_bin_best_minus_SP0_oracle_SF50": dense_area_bin.get("best_dense_minus_SP0_oracle_SF50"),
        "phase2_dense_area_bin_best_minus_SP0_majority_IoU": dense_area_bin.get("best_dense_minus_SP0_majority_IoU"),
        "phase2_sam2_filtered_minus_cropformer_gt_best_IoU_mean": sam2_source_filtered.get("sam2_minus_cropformer_gt_best_IoU_mean"),
        "phase2_sam2_filtered_minus_cropformer_source_oracle_SF50": sam2_source_filtered.get("sam2_minus_cropformer_source_oracle_SF50"),
        "phase2_sam2_relaxed_minus_cropformer_gt_best_IoU_mean": sam2_source_relaxed.get("sam2_minus_cropformer_gt_best_IoU_mean"),
        "phase2_sam2_relaxed_minus_cropformer_source_oracle_SF50": sam2_source_relaxed.get("sam2_minus_cropformer_source_oracle_SF50"),
        "phase2_sam2_relaxed_can_replace_v72_phase2_full_source": sam2_source_relaxed.get("can_replace_v72_phase2_full_source"),
        "phase3_d4rt_no_area_decision": d4rt_no_area.get("decision"),
        "phase3_d4rt_area_bin_decision": d4rt_area_bin.get("decision"),
        "phase4_no_area_best_variant": objectness_no_area.get("best_non_oracle_variant"),
        "phase4_no_area_SF50_top64": objectness_no_area.get("best_non_oracle_top64_iou50_rate"),
        "phase4_no_area_AUC": objectness_no_area.get("best_non_oracle_AUC_iou50"),
        "phase4_area_bin_best_variant": objectness_area_bin.get("best_non_oracle_variant"),
        "phase4_area_bin_SF50_top64": objectness_area_bin.get("best_non_oracle_top64_iou50_rate"),
        "phase4_area_bin_AUC": objectness_area_bin.get("best_non_oracle_AUC_iou50"),
        "phase3_no_area_subproposal_membership_from_carrier_uv": d4rt_no_area.get("subproposal_membership_from_carrier_uv"),
        "phase3_no_area_carrier_inside_ratio_mean": (d4rt_no_area.get("carrier_uv_membership_stats") or {}).get("proposal_inside_ratio_mean"),
        "phase3_no_area_hard_filter_background_delta": d4rt_no_area.get("hard_filter_background_false_positive_delta"),
        "phase3_area_bin_subproposal_membership_from_carrier_uv": d4rt_area_bin.get("subproposal_membership_from_carrier_uv"),
        "phase3_area_bin_carrier_inside_ratio_mean": (d4rt_area_bin.get("carrier_uv_membership_stats") or {}).get("proposal_inside_ratio_mean"),
        "phase3_area_bin_hard_filter_background_delta": d4rt_area_bin.get("hard_filter_background_false_positive_delta"),
        "phase5_best_variant": best_setcover.get("variant"),
        "phase5_best_atom_incidence_level": best_setcover.get("atom_incidence_level"),
        "phase5_best_SF50": best_setcover.get("representative_proposal_oracle_SF50_diagnostic"),
        "phase5_best_GT_best_IoU": best_setcover.get("representative_proposal_GT_best_IoU_mean_diagnostic"),
        "phase5_best_total_coverage": best_setcover.get("covered_total_atom_weight_ratio"),
        "phase5_best_D4RT_coverage": best_setcover.get("covered_D4RT_atom_weight_ratio"),
        "phase5_best_unresolved_broad_underseg_rate": best_setcover.get("unresolved_broad_underseg_rate"),
        "missing_summary_inputs": missing,
        "notes": [
            "Final decision is a continued No-Go. Phase4 objectness ranking has at least one passing branch, but Phase5 still fails after base set-cover, temporal prototype group repair, hard-risk group repair, no-GT member repair, PSC9/PSC10 group-mode fix, carrier-UV subproposal membership, and key-atom UV coverage repair.",
            "Targeted SAM2 source diagnostics did not outperform CropFormer on the current 20-frame v51 stacks, so direct replacement with existing SAM2 artifacts is not a supported repair.",
            "Carrier-UV and key-atom UV repairs remove the source-level-only D4RT limitation for the tested branches, but they do not solve object completeness, semantic/total coverage, or unresolved broad/underseg risk.",
            "No Phase6 local birth, local2history, or method success is claimed.",
        ],
    }
    _write_json(output_root / "final_decision.json", summary)
    _write_csv(output_root / "phase_summary_rows.csv", phase_rows)
    _write_csv(output_root / "decision_matrix_rows.csv", decision_matrix_rows)
    sha_rows = []
    for path in list(paths.values()) + sorted(output_root.glob("*")):
        if path.exists() and path.is_file() and path.name != "sha256_rows.csv":
            sha_rows.append({"path": _rel(path), "bytes": path.stat().st_size, "sha256": _sha256(path)})
    _write_csv(output_root / "sha256_rows.csv", sha_rows)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build v72 final decision from current phase artifacts.")
    parser.add_argument("--phase0-summary", default="outputs/audit/v72_phase0_fact_lock/fact_lock_summary.json")
    parser.add_argument("--phase1-summary", default="outputs/audit/v72_phase1_signal_adequacy/signal_adequacy_summary.json")
    parser.add_argument("--phase2-summary", default="outputs/audit/v72_phase2_semantic_proposals/semantic_proposal_summary.json")
    parser.add_argument("--phase2-casebook", default="outputs/audit/v72_visualizations/semantic_proposals_broad_cases/casebook_summary.json")
    parser.add_argument("--phase2-dense-no-area-floor-summary", default="outputs/audit/v72_phase2_dense_token_proposals_smoke10_hybrid_no_area_floor/dense_token_proposal_summary.json")
    parser.add_argument("--phase2-dense-area-bin-summary", default="outputs/audit/v72_phase2_dense_token_proposals_smoke10_area_bin1/dense_token_proposal_summary.json")
    parser.add_argument("--phase2-sam2-source-filtered-summary", default="outputs/audit/v72_phase2_sam2_source_diagnostic/sam2_source_diagnostic_summary.json")
    parser.add_argument("--phase2-sam2-source-relaxed-summary", default="outputs/audit/v72_phase2_sam2_source_diagnostic_relaxed/sam2_source_diagnostic_summary.json")
    parser.add_argument("--phase3-d4rt-no-area-floor-summary", default="outputs/audit/v72_phase3_d4rt_proposal_verification_no_area_floor_uvmember/d4rt_proposal_summary.json")
    parser.add_argument("--phase3-d4rt-area-bin-summary", default="outputs/audit/v72_phase3_d4rt_proposal_verification_area_bin1_uvmember/d4rt_proposal_summary.json")
    parser.add_argument("--phase4-objectness-no-area-floor-summary", default="outputs/audit/v72_phase4_objectness_ranking_no_area_floor_uvmember/objectness_summary.json")
    parser.add_argument("--phase4-objectness-area-bin-summary", default="outputs/audit/v72_phase4_objectness_ranking_area_bin1_uvmember/objectness_summary.json")
    parser.add_argument("--phase5-setcover-no-area-floor-summary", default="outputs/audit/v72_phase5_proposal_setcover_no_area_floor_uvmember_uvcoverage/proposal_setcover_summary.json")
    parser.add_argument("--phase5-setcover-area-bin-summary", default="outputs/audit/v72_phase5_proposal_setcover_area_bin1_uvmember_uvcoverage/proposal_setcover_summary.json")
    parser.add_argument("--output-root", default="outputs/audit/v72_final_decision")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
