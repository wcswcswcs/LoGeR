#!/usr/bin/env python3
"""Aggregate ACL2 v24 path-specific semantic short-rollout metrics."""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import v22_candidate_bank_report as impl  # noqa: E402
from tools import v18_true_action_report as base_impl  # noqa: E402


base_impl.FAMILY_BY_CANDIDATE.update(
    {
        "K1_H9": "baseline",
        "P0_01_SEMANTIC_ROLE_NOOP_IGNORED": "v24_phase0_noop",
        "P0_02_SEMANTIC_ROLE_PASS_THROUGH_CONSUMED": "v24_phase0_noop",
        "P0_03_SEMANTIC_ROLE_DEBUG_ONLY_ALL_MEMORY": "v24_phase0_debug",
        "PASSIVE_DEBUG_ONLY": "v24_passive_debug",
        "FRAMESEM_01_STRUCTURE_KEEP_LOWSTUFF_SOFT": "v24_single_frame_source",
        "FRAMESEM_02_LOWSTUFF_HIGHD_SKIP": "v24_single_frame_source",
        "FRAMESEM_03_SKY_NEUTRAL_VEGETATION_HIGHD_SKIP": "v24_single_frame_source_coarse_fallback",
        "GLOBALSEM_01_STRUCTURE_KEEP_LOWSTUFF_SOFT": "v24_single_global_source",
        "GLOBALSEM_02_LOWSTUFF_HIGHD_SKIP": "v24_single_global_source",
        "FRAMEGLOBAL_01_FRAME_ONLY": "v24_single_frame_source",
        "FRAMEGLOBAL_02_GLOBAL_ONLY": "v24_single_global_source",
        "FRAMEGLOBAL_03_FRAME_AND_GLOBAL": "v24_single_frame_global_source",
        "SWASEM_01_STRUCTURE_CACHE_KEEP": "v24_single_swa_cache",
        "SWASEM_02_LOWSTUFF_HIGHD_CACHE_SOFTDROP": "v24_single_swa_cache",
        "SWASEM_03_SKY_PROTECT_VEG_HIGHD_DROP": "v24_single_swa_cache_coarse_fallback",
        "SWASEM_04_PREVIOUS_SOURCE_ONLY": "v24_single_swa_cache",
        "SWASEM_05_OVERLAP_ONLY": "v24_single_swa_cache",
        "SWASEM_06_CURRENT_AND_PREVIOUS_COMPARE": "v24_single_swa_cache",
        "TTTSEM_01_STRUCTURE_POSITIVE_LONG": "v24_single_ttt_write",
        "TTTSEM_02_LOWSTUFF_HIGHD_NEGATIVE_SHORT": "v24_single_ttt_write",
        "TTTSEM_03_SKY_NEUTRAL_PROTECT": "v24_single_ttt_write_coarse_fallback",
        "TTTSEM_04_SEMANTIC_PLUS_TTT_CONFLICT": "v24_single_ttt_write",
        "TTTSEM_05_SEMANTIC_PLUS_DG_PLUS_CONFLICT": "v24_single_ttt_write",
        "TTTSEM_06_ROLE_SPECIFIC_BRANCH_W0": "v24_single_ttt_write",
        "TTTSEM_07_ROLE_SPECIFIC_LONG_SHORT": "v24_single_ttt_lifecycle",
        "CHUNKSEM_01_STRUCTURE_KEEP": "v24_single_global_source",
        "CHUNKSEM_02_LOWSTUFF_HIGHD_SKIP": "v24_single_global_source",
        "CHUNKSEM_03_PROTECT_SPECIAL_TOKENS": "v24_single_global_source",
        "PAIR_FRAME_TTT_PATHSPEC": "v24_pairwise_frame_ttt",
        "PAIR_FRAME_SWA_PATHSPEC": "v24_pairwise_frame_swa",
        "PAIR_SWA_TTT_PATHSPEC": "v24_pairwise_swa_ttt",
        "PAIR_GLOBAL_TTT_PATHSPEC": "v24_pairwise_global_ttt",
        "PAIR_FRAME_GLOBAL_PATHSPEC": "v24_pairwise_frame_global",
        "PAIR_FRAME_GLOBAL_SWA_TTT_PATHSPEC": "v24_pairwise_all",
        "ALLMEM_01_FRAME_TTT_PATHSPEC": "v24_allmem_pathspec",
        "ALLMEM_02_FRAME_SWA_TTT_PATHSPEC": "v24_allmem_pathspec",
        "ALLMEM_03_FRAME_GLOBAL_SWA_TTT_PATHSPEC": "v24_allmem_pathspec",
        "ALLMEM_04_SKY_NEUTRAL_STRUCTURE_LONG": "v24_allmem_pathspec_coarse_fallback",
        "ALLMEM_05_LOWSTUFF_HIGHD_SHORTNEG": "v24_allmem_pathspec",
        "ALLMEM_06_CONFLICT_GATED_SEMANTIC": "v24_allmem_pathspec",
        "FG_FINE_01_STRUCTURE_KEEP": "v26_single_frame_global_fine",
        "FG_FINE_02_LOWSTUFF_HIGHD_SKIP": "v26_single_frame_global_fine",
        "FG_FINE_03_SKY_NEUTRAL": "v26_single_frame_global_fine",
        "FG_FINE_04_STRUCTURE_RESCUE": "v26_single_frame_global_fine",
        "FG_FINE_05_CONFLICT_CONDITIONED": "v26_single_frame_global_fine",
        "SWA_FINE_01_OVERLAP_STRUCTURE_KEEP": "v26_single_swa_fine",
        "SWA_FINE_02_SKY_PARTIAL_KEEP": "v26_single_swa_fine",
        "SWA_FINE_03_VEGETATION_CONDITIONAL": "v26_single_swa_fine",
        "SWA_FINE_04_BOUNDARY_PROTECT": "v26_single_swa_fine",
        "SWA_FINE_05_CACHE_LIFECYCLE": "v26_single_swa_fine",
        "TTT_FINE_01_STRUCTURE_POSITIVE": "v26_single_ttt_fine",
        "TTT_FINE_02_SKY_NEUTRAL": "v26_single_ttt_fine",
        "TTT_FINE_03_SCALE_CONDITIONED": "v26_single_ttt_fine",
        "TTT_FINE_04_LOWSTUFF_HIGHD_SHORT": "v26_single_ttt_fine",
        "TTT_FINE_05_STRUCTURE_PROTECT": "v26_single_ttt_fine",
        "TTT_FINE_RISK_01_CONFLICT_TRI": "v26_ttt_fine_conflict_risk",
        "TTT_FINE_RISK_02_SCALE_STATE": "v26_ttt_fine_scale_risk",
        "TTT_FINE_RISK_03_CONFLICT_COMMIT_FILTER": "v26_ttt_fine_conflict_commit_filter",
        "TTT_FINE_REPAIR_01_SCALE_DUAL_LIFETIME": "v26_ttt_fine_scale_dual_lifetime",
        "TTT_FINE_REPAIR_02_SCALE_CONFLICT_COMMIT_FILTER": "v26_ttt_fine_scale_conflict_commit_filter",
        "FG_RISK_00": "v27_phase2_fg_causal",
        "FG_SEM_01": "v27_phase2_fg_causal",
        "FG_SEM_02": "v27_phase2_fg_causal",
        "FG_SEM_03": "v27_phase2_fg_causal",
        "FG_SEM_04": "v27_phase2_fg_causal",
        "FG_SEM_05": "v27_phase2_fg_causal",
        "SWA_SEM_01": "v27_phase2_swa_causal",
        "SWA_SEM_02": "v27_phase2_swa_causal",
        "SWA_SEM_03": "v27_phase2_swa_causal",
        "SWA_SEM_04": "v27_phase2_swa_causal",
        "SWA_SEM_05": "v27_phase2_swa_causal",
        "TTT_ROLE_00_RISK_ONLY": "v27_phase2_ttt_causal",
        "TTT_ROLE_01_STRUCTURE_LOWCONFLICT_POS": "v27_phase2_ttt_causal",
        "TTT_ROLE_02_LOWSTUFF_HIGHD_CONFLICT_SHORTNEG": "v27_phase2_ttt_causal",
        "TTT_ROLE_03_VEGETATION_CONDITIONAL_NEG": "v27_phase2_ttt_causal",
        "TTT_ROLE_04_BLOCK_HIGHCONFLICT_STRUCTURE_LONGWRITE": "v27_phase2_ttt_causal",
        "TTT_ROLE_05_FULL_ROLE_TREE": "v27_phase2_ttt_causal",
        "FRAME_SEM_ONLY": "v28_frame_semantic_only",
        "FRAME_RISK_ONLY": "v28_frame_risk_only",
        "FRAME_SEM_RISK": "v28_frame_semantic_risk",
        "GLOBAL_SEM_ONLY": "v28_global_semantic_only",
        "GLOBAL_RISK_ONLY": "v28_global_risk_only",
        "GLOBAL_SEM_RISK": "v28_global_semantic_risk",
        "SWA_SEM_ONLY": "v28_swa_semantic_only",
        "SWA_RISK_ONLY": "v28_swa_risk_only",
        "SWA_SEM_RISK": "v28_swa_semantic_risk",
        "TTT_SEM_ONLY": "v28_ttt_semantic_only",
        "TTT_RISK_ONLY": "v28_ttt_risk_only",
        "TTT_SEM_RISK": "v28_ttt_semantic_risk",
        "V29C_BASE_H9_REFERENCE": "v29c_causal_bank_reference",
        "V29C_CAUSAL_FRAME_SKIP_TOP": "v29c_masklet_causal_frame",
        "V29C_CAUSAL_GLOBAL_SKIP_TOP": "v29c_masklet_causal_global",
        "V29C_CAUSAL_SWA_ANCHOR_TOP": "v29c_masklet_causal_swa",
        "V29C_CAUSAL_SWA_REMOVE_TOP": "v29c_masklet_causal_swa",
        "V29C_CAUSAL_TTT_POS_TOP": "v29c_masklet_causal_ttt",
        "V29C_CAUSAL_TTT_NEG_TOP": "v29c_masklet_causal_ttt",
        "V30_BASE_H9_REFERENCE": "v30_causal_bank_reference",
        "V30_MASKLET_FRAME_SKIP": "v30_masklet_causal_frame",
        "V30_MASKLET_GLOBAL_SKIP": "v30_masklet_causal_global",
        "V30_MASKLET_SWA_ANCHOR": "v30_masklet_causal_swa",
        "V30_MASKLET_SWA_REMOVE": "v30_masklet_causal_swa",
        "V30_MASKLET_TTT_POS": "v30_masklet_causal_ttt",
        "V30_MASKLET_TTT_NEG": "v30_masklet_causal_ttt",
        "V31_BASE_H9_REFERENCE": "v31_h9_reference",
        "V31_A0_ORIG_C23": "v31_track_a_original_c23",
        "V31_A1_SEM_Z_FINE": "v31_track_a_semantic_z_fine",
        "V31_A1B_SEM_Z_COARSE": "v31_track_a_semantic_z_coarse",
        "V31_A5_SEM_RESID_FINE_L025": "v31_track_a_semantic_residual_fine",
        "V31_A5B_SEM_RESID_COARSE_L025": "v31_track_a_semantic_residual_coarse",
        "V31_B0_STATIC_RESCUE_EXISTING": "v31_track_b_existing_static_rescue",
    }
)


def _arg_value(name: str) -> str | None:
    if name not in sys.argv:
        return None
    idx = sys.argv.index(name)
    if idx + 1 >= len(sys.argv):
        return None
    return sys.argv[idx + 1]


def _read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _to_float(value: object) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return float("nan")


def _postprocess_v24(out_dir: Path) -> None:
    md_old = out_dir / "acl2_v18_true_action_report.md"
    md_new = out_dir / "acl2_v24_candidate_bank_report.md"
    if md_old.exists():
        text = md_old.read_text(encoding="utf-8")
        text = text.replace(
            "# ACL2 v18 True Action Candidate Report",
            "# ACL2 v24 Path-Specific Semantic Candidate Report",
        )
        text = text.replace("v18 true-action", "v24 path-specific semantic")
        md_new.write_text(text, encoding="utf-8")

    rows = _read_csv(out_dir / "candidate_vs_H9_delta_by_horizon.csv")
    grouped: Dict[tuple[str, int], Dict[int, Dict[str, str]]] = {}
    for row in rows:
        candidate = row.get("candidate_id", "")
        if candidate == "K1_H9":
            continue
        try:
            chunk = int(row.get("chunk_id", ""))
            horizon = int(row.get("horizon", ""))
        except ValueError:
            continue
        grouped.setdefault((candidate, chunk), {})[horizon] = row

    gate_rows: List[Dict[str, object]] = []
    for (candidate, chunk), by_h in sorted(grouped.items()):
        h10 = by_h.get(10)
        h15 = by_h.get(15)
        h10_ate = _to_float(h10.get("ATE_delta_vs_H9")) if h10 else float("nan")
        h15_ate = _to_float(h15.get("ATE_delta_vs_H9")) if h15 else float("nan")
        h10_seg = _to_float(h10.get("intersection_200_300_delta_vs_H9")) if h10 else float("nan")
        h15_seg = _to_float(h15.get("intersection_200_300_delta_vs_H9")) if h15 else float("nan")
        h10_down = _to_float(h10.get("intersection_400_600_delta_vs_H9")) if h10 else float("nan")
        h15_down = _to_float(h15.get("intersection_400_600_delta_vs_H9")) if h15 else float("nan")
        durability = (
            max(0.0, -h15_ate) / max(1e-8, max(0.0, -h10_ate))
            if math.isfinite(h10_ate) and math.isfinite(h15_ate)
            else float("nan")
        )
        phase2_single_gate = (
            (math.isfinite(h10_seg) and h10_seg <= -3.0)
            or (math.isfinite(h15_ate) and h15_ate <= -1.5)
            or (math.isfinite(h15_seg) and h15_seg <= -2.5)
        )
        phase2_downstream_ok = (
            (not math.isfinite(h10_down) or h10_down <= 1.0)
            and (not math.isfinite(h15_down) or h15_down <= 1.0)
        )
        phase34_local_gate = (
            (math.isfinite(h10_ate) and h10_ate <= -3.0)
            or (math.isfinite(h15_ate) and h15_ate <= -3.0)
            or (math.isfinite(h10_seg) and h10_seg <= -5.0)
            or (math.isfinite(h15_seg) and h15_seg <= -5.0)
        )
        phase3_gate = bool(phase34_local_gate and phase2_downstream_ok and math.isfinite(durability) and durability >= 0.35)
        phase4_gate = bool(phase34_local_gate and phase2_downstream_ok and math.isfinite(durability) and durability >= 0.45)
        gate_rows.append(
            {
                "candidate_id": candidate,
                "family": base_impl.FAMILY_BY_CANDIDATE.get(candidate, "unknown"),
                "chunk_id": chunk,
                "h10_ATE_delta_vs_H9": h10_ate,
                "h15_ATE_delta_vs_H9": h15_ate,
                "h10_200_300_delta_vs_H9": h10_seg,
                "h15_200_300_delta_vs_H9": h15_seg,
                "h10_400_600_delta_vs_H9": h10_down,
                "h15_400_600_delta_vs_H9": h15_down,
                "durability_ratio": durability,
                "phase2_single_path_gate_pass": bool(phase2_single_gate and phase2_downstream_ok),
                "phase3_pairwise_gate_pass": phase3_gate,
                "phase4_allmem_gate_pass": phase4_gate,
                "selector_allowed": phase4_gate,
                "counts_as_online_ttt_write_success": False,
            }
        )

    _write_csv(out_dir / "v24_gate_by_candidate_chunk.csv", gate_rows)
    summary = {
        "num_candidate_chunks": len(gate_rows),
        "phase2_gate_pass_candidates": sorted(
            {str(row["candidate_id"]) for row in gate_rows if row["phase2_single_path_gate_pass"]}
        ),
        "phase3_gate_pass_candidates": sorted(
            {str(row["candidate_id"]) for row in gate_rows if row["phase3_pairwise_gate_pass"]}
        ),
        "phase4_gate_pass_candidates": sorted(
            {str(row["candidate_id"]) for row in gate_rows if row["phase4_allmem_gate_pass"]}
        ),
        "selector_allowed": any(bool(row["selector_allowed"]) for row in gate_rows),
        "full_online_validation_allowed": False,
    }
    (out_dir / "v24_gate_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if md_new.exists():
        md_new.write_text(
            md_new.read_text(encoding="utf-8")
            + "\n## v24 Gate Overlay\n\n"
            + f"Phase 2 pass candidates: `{summary['phase2_gate_pass_candidates']}`\n\n"
            + f"Phase 3 pass candidates: `{summary['phase3_gate_pass_candidates']}`\n\n"
            + f"Phase 4 pass candidates: `{summary['phase4_gate_pass_candidates']}`\n",
            encoding="utf-8",
        )


def main() -> None:
    out_dir_text = _arg_value("--out-dir")
    impl.main()
    if out_dir_text:
        _postprocess_v24(Path(out_dir_text))


if __name__ == "__main__":
    main()
