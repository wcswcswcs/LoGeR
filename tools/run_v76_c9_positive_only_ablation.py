#!/usr/bin/env python3
"""Build ACL2 v76-TF Phase 1 positive-only ablation tables."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any, Dict, List, Mapping, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.v76tf_common import (
V45_CLEAN_REGISTRY,
    V45_INTERACTION,
    V45_LEDGER,
    V46B_REGISTRY,
    V76_ROOT,
    boolish,
    ensure_dir,
    first_row,
    read_csv,
    rel,
    safe_float,
    write_csv,
    write_json,
    write_text,
)


FACTORIAL_ROWS = [
    "F000_NONE",
    "F100_ONLY_FRAME_ATTN",
    "F010_ONLY_TTT",
    "F001_ONLY_SWA",
    "F110_FRAME_ATTN_TTT",
    "F101_FRAME_ATTN_SWA",
    "F011_TTT_SWA",
    "F111_ALL_THREE",
]

PLAN_CANDIDATE_MAP = {
    "START_A_CLEAN_H35": "F000_NONE",
    "H35_PLUS_READ_MAP": "F100_ONLY_FRAME_ATTN",
    "H35_PLUS_TTT_TRI_REPLAY": "F010_ONLY_TTT",
    "H35_PLUS_SWA_REPLACE": "F001_ONLY_SWA",
    "H35_PLUS_READ_TTT": "F110_FRAME_ATTN_TTT",
    "H35_PLUS_READ_SWA": "F101_FRAME_ATTN_SWA",
    "H35_PLUS_TTT_SWA": "F011_TTT_SWA",
    "H35_PLUS_READ_TTT_SWA": "F111_ALL_THREE",
}

START_B_SEM_READ_REGISTRY = (
    V45_CLEAN_REGISTRY.parents[3]
    / "phase5_semantic_read/report_R1/sem2_vs_c9clean/full_online_registry.csv"
)

D_ROW_DESCRIPTIONS = {
    "F0": "Exact C9 repeat.",
    "D1": "Fixed read beta substitute; C9 read beta chunk map removed.",
    "D2": "Fixed tri gamma 0.003 substitute; C9 tri gamma map/tri replay chunk params removed.",
    "D3": "Fixed tri gamma 0.004 substitute; best fixed tri gamma substitute.",
    "D4": "Fixed tri gamma 0.005 substitute.",
    "D5": "Commit EMA off.",
    "D6": "Global commit EMA alpha 0.8 on branch 0.",
    "D7": "C9-clean fixed substitute row; removes chunk-id policies but is not a clean positive-only start-B factorial.",
}


def _copy_factorial_rows() -> List[Dict[str, Any]]:
    registry = read_csv(V46B_REGISTRY)
    by_row = {str(row.get("row")): row for row in registry}
    out: List[Dict[str, Any]] = []
    f000 = first_row(registry, "row", "F000_NONE")
    base_ate = safe_float(f000.get("ATE_full")) if f000 else None
    for row_name in FACTORIAL_ROWS:
        row = by_row.get(row_name)
        if not row:
            out.append({
                "row": row_name,
                "available": False,
                "source_artifact": rel(V46B_REGISTRY),
            })
            continue
        ate = safe_float(row.get("ATE_full"))
        gain = base_ate - ate if base_ate is not None and ate is not None else None
        out.append({
            "row": row_name,
            "available": True,
            "source_artifact": rel(V46B_REGISTRY),
            "run_name": row.get("run_name"),
            "FRAME_ATTN": row.get("FRAME_ATTN"),
            "TTT": row.get("TTT"),
            "SWA": row.get("SWA"),
            "status": row.get("status"),
            "frames": row.get("frames"),
            "ATE_full": row.get("ATE_full"),
            "gain_vs_F000": gain,
            "Rot_full": row.get("Rot_full"),
            "FinalErr_full": row.get("FinalErr_full"),
            "segment_200_300_ATE": row.get("segment_200_300_ATE"),
            "segment_400_600_ATE": row.get("segment_400_600_ATE"),
            "hmc_rows": row.get("hmc_rows"),
            "frame_attn_read_control_active": row.get("frame_attn_read_control_active"),
            "ttt_tri_replay_applied_count": row.get("ttt_tri_replay_applied_count"),
            "ttt_tri_replay_applied_layer_count_sum": row.get("ttt_tri_replay_applied_layer_count_sum"),
            "ttt_positive_mass_mean": row.get("ttt_positive_mass_mean"),
            "ttt_neutral_mass_mean": row.get("ttt_neutral_mass_mean"),
            "ttt_negative_mass_mean": row.get("ttt_negative_mass_mean"),
            "swa_overlap_replace_applied_count": row.get("swa_overlap_replace_applied_count"),
            "no_chunk_policy_pass": row.get("no_chunk_policy_pass"),
            "row_valid": row.get("row_valid"),
            "invalid_reason": row.get("invalid_reason"),
        })
    return out


def _plan_candidate_rows(factorial_rows: List[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    by_row = {str(row.get("row")): row for row in factorial_rows}
    out: List[Dict[str, Any]] = []
    for candidate, row_name in PLAN_CANDIDATE_MAP.items():
        row = by_row.get(row_name, {})
        out.append({
            "plan_candidate": candidate,
            "evidence_row": row_name,
            "start": "clean_H35",
            "available": bool(row.get("available")),
            "ATE_full": row.get("ATE_full"),
            "gain_vs_start": row.get("gain_vs_F000"),
            "source_artifact": row.get("source_artifact"),
            "training_free_boundary": "clean no-chunk v46B factorial",
        })
    start_b_rows = read_csv(START_B_SEM_READ_REGISTRY)
    d7 = first_row(start_b_rows, "name", "D7")
    sem2 = first_row(start_b_rows, "name", "SEM2")
    d7_ate = safe_float(d7.get("ATE_full")) if d7 else None
    sem2_ate = safe_float(sem2.get("ATE_full")) if sem2 else None
    sem2_gain = d7_ate - sem2_ate if d7_ate is not None and sem2_ate is not None else None
    out.append({
        "plan_candidate": "START_B_C9CLEAN_DECHUNK",
        "evidence_row": "D7",
        "start": "C9_clean_dechunk",
        "available": d7 is not None,
        "ATE_full": d7.get("ATE_full") if d7 else "",
        "gain_vs_start": 0.0 if d7 else "",
        "source_artifact": rel(START_B_SEM_READ_REGISTRY),
        "training_free_boundary": "D7 C9-clean fixed substitute; no chunk maps in candidate chunk audit",
    })
    out.append({
        "plan_candidate": "C9CLEAN_PLUS_SEM_RESID_READ_ONLY",
        "evidence_row": "SEM2",
        "start": "C9_clean_dechunk",
        "available": sem2 is not None,
        "ATE_full": sem2.get("ATE_full") if sem2 else "",
        "gain_vs_start": sem2_gain if sem2 else "",
        "source_artifact": rel(START_B_SEM_READ_REGISTRY),
        "training_free_boundary": "full-run semantic residual READ-only evidence on C9-clean; not a C9 read-beta-map restoration",
    })
    for candidate in (
        "C9CLEAN_PLUS_READ_BETA_MAP",
        "C9CLEAN_PLUS_TTT_TRI_REPLAY",
        "C9CLEAN_PLUS_COMMIT_EMA",
        "C9CLEAN_PLUS_SWA_REPLACE",
        "C9CLEAN_PLUS_READ_TTT_COMMIT",
    ):
        out.append({
            "plan_candidate": candidate,
            "evidence_row": "",
            "start": "C9_clean_dechunk",
            "available": False,
            "ATE_full": "",
            "gain_vs_start": "",
            "source_artifact": "",
            "training_free_boundary": "missing exact start-B positive-only factorial artifact; v45 D rows are C9 knockout/substitute evidence only",
        })
    return out


def _c9_clean_rows() -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in read_csv(V45_CLEAN_REGISTRY):
        name = str(row.get("name") or "")
        out.append({
            "name": name,
            "description": D_ROW_DESCRIPTIONS.get(name, ""),
            "source_artifact": rel(V45_CLEAN_REGISTRY),
            "status": row.get("status"),
            "frames": row.get("frames"),
            "ATE_full": row.get("ATE_full"),
            "delta_vs_historical_c9_ATE": row.get("delta_vs_historical_c9_ATE"),
            "segment_200_300_ATE": row.get("segment_200_300_ATE"),
            "segment_400_600_ATE": row.get("segment_400_600_ATE"),
            "hmc_rows": row.get("hmc_rows"),
            "evidence_kind": "exact_C9_knockout_or_substitute_not_startB_positive_only",
        })
    return out


def _ledger_rows() -> List[Dict[str, Any]]:
    return [
        {
            "component": row.get("component"),
            "effect_delta_vs_C9": row.get("effect_delta_vs_C9"),
            "source_artifact": rel(V45_LEDGER),
            "evidence_kind": "exact_C9_knockout_or_substitute",
        }
        for row in read_csv(V45_LEDGER)
    ]


def _summary(factorial_rows: List[Mapping[str, Any]], candidate_rows: List[Mapping[str, Any]]) -> Dict[str, Any]:
    valid_rows = [
        row for row in factorial_rows
        if row.get("available") and boolish(row.get("row_valid")) and boolish(row.get("no_chunk_policy_pass"))
    ]
    gains = [safe_float(row.get("gain_vs_F000")) for row in valid_rows if str(row.get("row")) != "F000_NONE"]
    gains = [value for value in gains if value is not None]
    by_row = {str(row.get("row")): row for row in factorial_rows}
    def gain(name: str) -> Optional[float]:
        return safe_float(by_row.get(name, {}).get("gain_vs_F000"))
    read_gain = gain("F100_ONLY_FRAME_ATTN")
    ttt_gain = gain("F010_ONLY_TTT")
    swa_gain = gain("F001_ONLY_SWA")
    read_ttt_gain = gain("F110_FRAME_ATTN_TTT")
    all_three_gain = gain("F111_ALL_THREE")
    read_ttt_synergy = None
    if read_ttt_gain is not None and read_gain is not None and ttt_gain is not None:
        read_ttt_synergy = read_ttt_gain - max(read_gain, ttt_gain)
    start_b_gains = [
        safe_float(row.get("gain_vs_start"))
        for row in candidate_rows
        if row.get("start") == "C9_clean_dechunk" and bool(row.get("available")) and str(row.get("plan_candidate")) != "START_B_C9CLEAN_DECHUNK"
    ]
    start_b_gains = [value for value in start_b_gains if value is not None]
    start_b_available = any(
        row.get("start") == "C9_clean_dechunk" and bool(row.get("available"))
        for row in candidate_rows
    )
    start_b_positive = bool(start_b_gains and max(start_b_gains) >= 0.3)
    return {
        "source_artifacts": {
            "start_a_clean_h35_factorial": rel(V46B_REGISTRY),
            "c9_clean_rows": rel(V45_CLEAN_REGISTRY),
            "start_b_semantic_read_registry": rel(START_B_SEM_READ_REGISTRY),
            "c9_component_ledger": rel(V45_LEDGER),
            "c9_interaction_matrix": rel(V45_INTERACTION),
        },
        "required_factorial_rows": FACTORIAL_ROWS,
        "available_factorial_rows": [str(row.get("row")) for row in factorial_rows if row.get("available")],
        "all_start_a_rows_valid_no_chunk": len(valid_rows) == len(FACTORIAL_ROWS),
        "start_a_positive_signal": bool(gains and max(gains) >= 0.3),
        "start_a_best_gain_vs_F000": max(gains) if gains else None,
        "read_only_gain_vs_F000": read_gain,
        "ttt_only_gain_vs_F000": ttt_gain,
        "swa_only_gain_vs_F000": swa_gain,
        "read_ttt_gain_vs_F000": read_ttt_gain,
        "read_ttt_synergy_over_best_single": read_ttt_synergy,
        "all_three_gain_vs_F000": all_three_gain,
        "ttt_tri_replay_applied_count_F010": by_row.get("F010_ONLY_TTT", {}).get("ttt_tri_replay_applied_count"),
        "ttt_tri_replay_layer_count_F010": by_row.get("F010_ONLY_TTT", {}).get("ttt_tri_replay_applied_layer_count_sum"),
        "start_b_positive_only_available": start_b_available,
        "start_b_positive_signal": start_b_positive,
        "start_b_best_gain_vs_D7": max(start_b_gains) if start_b_gains else None,
        "start_b_evidence_caveat": "Start B positive evidence is semantic residual READ-only; no Start B TTT/commit/SWA isolated full-run artifact was found.",
        "phase1_strict_plan_gate_pass": bool(gains and max(gains) >= 0.3 and start_b_positive),
        "phase1_start_a_signal_gate_pass": bool(gains and max(gains) >= 0.3),
        "blocker_if_strict_gate_fails": "Need both Start A and Start B positive signals; Start B only has semantic READ evidence, not all isolated C9 components.",
    }


def _write_report(out_dir: Path, summary: Mapping[str, Any], factorial_rows: List[Mapping[str, Any]]) -> None:
    lines = [
        "# v76 Phase 1 Positive-Only Ablation",
        "",
        "This report separates clean-H35 positive-only rows from exact C9 knockout/substitute evidence.",
        "",
        "## Start A Clean-H35 Factorial",
        "",
        "| row | ATE_full | gain_vs_F000 | read | ttt | swa | valid | ttt_applied |",
        "|---|---:|---:|---:|---:|---:|---|---:|",
    ]
    for row in factorial_rows:
        lines.append(
            f"| `{row.get('row')}` | {row.get('ATE_full')} | {row.get('gain_vs_F000')} | "
            f"{row.get('FRAME_ATTN')} | {row.get('TTT')} | {row.get('SWA')} | "
            f"{row.get('row_valid')}/no_chunk={row.get('no_chunk_policy_pass')} | "
            f"{row.get('ttt_tri_replay_applied_count')} |"
        )
    lines.extend([
        "",
        "## Gate",
        "",
        f"- Start A signal gate: `{summary['phase1_start_a_signal_gate_pass']}`.",
        f"- Strict plan gate: `{summary['phase1_strict_plan_gate_pass']}`.",
        f"- Start B positive-only available: `{summary['start_b_positive_only_available']}`.",
        f"- Start B best gain vs D7: `{summary['start_b_best_gain_vs_D7']}` m.",
        f"- Start B caveat: {summary['start_b_evidence_caveat']}",
        f"- Blocker if strict gate fails: {summary['blocker_if_strict_gate_fails']}",
        "",
        "## Interpretation",
        "",
        f"- READ-only gain vs F000: `{summary['read_only_gain_vs_F000']}` m.",
        f"- TTT-only gain vs F000: `{summary['ttt_only_gain_vs_F000']}` m.",
        f"- SWA-only gain vs F000: `{summary['swa_only_gain_vs_F000']}` m.",
        f"- READ+TTT gain vs F000: `{summary['read_ttt_gain_vs_F000']}` m.",
        f"- READ+TTT synergy over best single: `{summary['read_ttt_synergy_over_best_single']}` m.",
        "",
        "Start A has a real positive-only signal. The strict Phase 1 plan gate remains partial until a true Start B C9-clean/dechunk positive-only factorial exists.",
    ])
    write_text(out_dir / "phase1_positive_only_report.md", "\n".join(lines) + "\n")


def run(out_dir: Path) -> Dict[str, Any]:
    ensure_dir(out_dir)
    factorial_rows = _copy_factorial_rows()
    candidate_rows = _plan_candidate_rows(factorial_rows)
    c9_clean = _c9_clean_rows()
    ledger = _ledger_rows()
    summary = _summary(factorial_rows, candidate_rows)
    write_csv(out_dir / "positive_only_factorial_table.csv", factorial_rows)
    write_csv(out_dir / "plan_candidate_evidence_table.csv", candidate_rows)
    write_csv(out_dir / "c9clean_substitute_rows_not_positive_only.csv", c9_clean)
    write_csv(out_dir / "exact_c9_knockout_table.csv", ledger)
    if V45_INTERACTION.exists():
        write_csv(out_dir / "exact_c9_interaction_table.csv", read_csv(V45_INTERACTION))
    write_json(out_dir / "phase1_positive_only_summary.json", summary)
    _write_report(out_dir, summary, factorial_rows)
    return {"out_dir": rel(out_dir), **summary}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=str(V76_ROOT / "phase1_positive_only_ablation"))
    parser.add_argument("--strict", action="store_true", help="Return nonzero if the strict Start A + Start B plan gate is partial.")
    args = parser.parse_args()
    result = run(Path(args.out_dir))
    write_json(Path(args.out_dir) / "command_result.json", result)
    if args.strict and not result["phase1_strict_plan_gate_pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
