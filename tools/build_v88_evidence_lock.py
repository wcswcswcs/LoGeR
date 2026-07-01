#!/usr/bin/env python3
"""Build ACL2 v88 Phase0 evidence lock from v87 artifacts."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

from v86_soft_latent_utils import read_json, safe_int, seq_norm, write_csv, write_json


DEFAULT_ROOT = Path("results/acl2_v88tf_scale_mode_consensus_gauge_update_attribution")
DEFAULT_OUT = DEFAULT_ROOT / "phase0_evidence_lock"

REQUIRED_INPUTS = [
    ("v87_plan", "docs/ACL2_v88TF_ScaleModeConsensus_GaugeUpdateAttribution_ExperimentPlan.md", "Phase0"),
    ("v87_execution_log", "docs/ACL2_v87TF_ScaleConditionedLatentGaugeCarrier_执行日志.md", "Phase0"),
    ("v87_recap_log", "docs/ACL2_v87TF_ScaleConditionedLatentGaugeCarrier_实验结果复盘.md", "Phase0"),
    (
        "v87_final_decision",
        "results/acl2_v87tf_scale_conditioned_latent_gauge_carrier/report_final/final_decision.json",
        "Phase0",
    ),
    (
        "v87_phase8_direct_pair_summary",
        "results/acl2_v87tf_scale_conditioned_latent_gauge_carrier/phase8_merge_gauge_direct_pair_weighting/merge_gauge_direct_pair_summary.json",
        "Phase0",
    ),
    (
        "v87_phase2_highobs_summary",
        "results/acl2_v87tf_scale_conditioned_latent_gauge_carrier/phase2_scale_relevance_k16_r1_median_abs_highobs/proxy_relevance_summary.json",
        "Phase0",
    ),
    (
        "v87_selected_phase1_by_adjacent",
        "results/acl2_v87tf_scale_conditioned_latent_gauge_carrier/phase1_scale_conditioned_pair_universe_k16_r1_median_abs/scale_conditioned_pair_by_adjacent.csv",
        "Phase1",
    ),
    (
        "v87_selected_phase1_rows",
        "results/acl2_v87tf_scale_conditioned_latent_gauge_carrier/phase1_scale_conditioned_pair_universe_k16_r1_median_abs/scale_conditioned_pair_rows.csv",
        "Phase1",
    ),
    (
        "v85_anchor_pair_rows",
        "results/acl2_v85tf_latent_anchor_alignment_pairwise_memory_ruler/phase1_anchor_pair_universe/anchor_pair_rows.csv",
        "Phase1",
    ),
]

FORBIDDEN_REPEATS = [
    "v84 source-side anchor mask boost",
    "v84 support-map-driven merge/gauge fallback",
    "v85 hard-anchor sufficiency threshold sweep",
    "v86 ridge lambda / feature dim micro sweep without new scale-mode hypothesis",
    "v87 CONFLICT-state threshold micro sweep",
    "v87 raw-overlap full_direct_pair weighting rerun as the main method",
    "running QK runtime action without Phase2/3/4/5 gates",
    "using pooled Q/K C as per-head route carrier",
    "TTT before SWA or merge/gauge confirmed evidence",
    "per-chunk Sim(3) runtime correction or direct scale multiplication",
    "GT-selected runtime threshold/head/layer",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def _load_json(path: str) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {}
    return read_json(p)


def _pair_key(row: dict[str, str]) -> tuple[str, int, int]:
    return (
        seq_norm(row.get("seq")),
        int(safe_int(row.get("prev_chunk")) or 0),
        int(safe_int(row.get("curr_chunk")) or 0),
    )


def _read_pair_keys(path: Path) -> set[tuple[str, int, int]]:
    keys: set[tuple[str, int, int]] = set()
    if not path.exists():
        return keys
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            keys.add(_pair_key(row))
    return keys


def _anchor_source_check(anchor_path: Path, requested_pairs: set[tuple[str, int, int]]) -> dict[str, Any]:
    source_by_pair: dict[tuple[str, int, int], set[str]] = {key: set() for key in requested_pairs}
    raw_path_count = 0
    existing_raw_path_count = 0
    missing_examples: list[str] = []
    if not anchor_path.exists():
        return {
            "requested_pair_count": len(requested_pairs),
            "pairs_with_source_path": 0,
            "raw_source_path_count": 0,
            "existing_raw_source_path_count": 0,
            "missing_raw_source_path_count": 0,
            "missing_raw_source_examples": [],
            "pairs_without_source_path": sorted(f"{a}:{b}->{c}" for a, b, c in requested_pairs)[:20],
        }
    with anchor_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            key = _pair_key(row)
            if key not in source_by_pair:
                continue
            source_path = (row.get("source_path") or "").strip()
            if not source_path:
                continue
            if source_path not in source_by_pair[key]:
                raw_path_count += 1
                if Path(source_path).exists():
                    existing_raw_path_count += 1
                elif len(missing_examples) < 10:
                    missing_examples.append(source_path)
            source_by_pair[key].add(source_path)
    pairs_without = [key for key, paths in source_by_pair.items() if not paths]
    return {
        "requested_pair_count": len(requested_pairs),
        "pairs_with_source_path": len(requested_pairs) - len(pairs_without),
        "raw_source_path_count": raw_path_count,
        "existing_raw_source_path_count": existing_raw_path_count,
        "missing_raw_source_path_count": raw_path_count - existing_raw_path_count,
        "missing_raw_source_examples": missing_examples,
        "pairs_without_source_path": sorted(f"{a}:{b}->{c}" for a, b, c in pairs_without)[:20],
    }


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    path_by_name = {name: path for name, path, _ in REQUIRED_INPUTS}
    required_rows = [
        {
            "name": name,
            "path": path,
            "exists": Path(path).exists(),
            "required_for": phase,
        }
        for name, path, phase in REQUIRED_INPUTS
    ]
    write_csv(args.out_dir / "required_inputs.csv", required_rows)
    write_csv(args.out_dir / "forbidden_repeats.csv", [{"forbidden_repeat": item} for item in FORBIDDEN_REPEATS])
    (args.out_dir / "forbidden_repeats.md").write_text(
        "# Forbidden Repeats\n\n" + "\n".join(f"- {item}" for item in FORBIDDEN_REPEATS) + "\n",
        encoding="utf-8",
    )

    final = _load_json(path_by_name["v87_final_decision"])
    phase8 = _load_json(path_by_name["v87_phase8_direct_pair_summary"])
    phase2 = _load_json(path_by_name["v87_phase2_highobs_summary"])
    metrics = final.get("key_metrics") or {}

    pair_keys = _read_pair_keys(Path(path_by_name["v87_selected_phase1_by_adjacent"]))
    raw_check = _anchor_source_check(Path(path_by_name["v85_anchor_pair_rows"]), pair_keys)

    checks = {
        "v87_final_status_no_go_before_runtime": final.get("final_status") == "No-Go_before_runtime_action",
        "v87_runtime_and_ttt_blocked": final.get("runtime_action_allowed") is False and final.get("ttt_allowed") is False,
        "v87_phase2_highobs_scale_proxy_pass_locked": phase2.get("phase2_scale_proxy_gate_pass") is True
        and metrics.get("phase2_highobs_gate_pass") is True,
        "v87_semantic_signal_false_locked": phase2.get("semantic_signal_pass") is False
        and metrics.get("phase2_semantic_signal_pass") is False,
        "v87_phase3_valid_rows_zero_locked": metrics.get("phase3_valid_rows") == 0,
        "v87_phase4_good_fpr_one_locked": metrics.get("phase4_good_FPR") == 1.0,
        "v87_phase8_raw_overlap_available_locked": phase8.get("raw_overlap_geometry_counterfactual_available") is True
        and metrics.get("phase8_raw_overlap_geometry_counterfactual_available") is True,
        "v87_phase8_raw_overlap_gate_fail_locked": phase8.get("phase8_raw_overlap_geometry_gate_pass") is False
        and metrics.get("phase8_raw_overlap_gate_pass") is False,
        "v87_phase8_bad_improvement_failed_locked": (metrics.get("phase8_bad_raw_overlap_median_improvement_vs_native") or 0.0) < 0.10,
        "forbidden_repeats_non_empty": len(FORBIDDEN_REPEATS) > 0,
        "required_inputs_present": all(row["exists"] for row in required_rows),
        "v87_selected_pair_universe_nonempty": len(pair_keys) >= 49,
        "raw_source_recoverable_for_selected_pairs": raw_check["pairs_with_source_path"] == raw_check["requested_pair_count"],
        "raw_source_files_exist": raw_check["missing_raw_source_path_count"] == 0 and raw_check["existing_raw_source_path_count"] > 0,
    }
    gate_pass = all(bool(v) for v in checks.values())

    boundary_lines = [
        "# v87 No-Go Boundary for v88",
        "",
        f"- final_status: `{final.get('final_status')}`",
        f"- blocker: `{final.get('blocker')}`",
        f"- active_decision_labels: `{', '.join(final.get('active_decision_labels') or [])}`",
        f"- phase2_highobs_gate_pass: `{metrics.get('phase2_highobs_gate_pass')}`",
        f"- phase2_S_overlap_rho: `{metrics.get('phase2_S_overlap_rho')}`",
        f"- phase2_S_overlap_good_low_fpr: `{metrics.get('phase2_S_overlap_good_low_fpr')}`",
        f"- phase2_semantic_signal_pass: `{metrics.get('phase2_semantic_signal_pass')}`",
        f"- phase3_valid_rows: `{metrics.get('phase3_valid_rows')}`",
        f"- phase3_blocker: `{metrics.get('phase3_blocker')}`",
        f"- phase4_good_FPR: `{metrics.get('phase4_good_FPR')}`",
        f"- phase8_raw_overlap_geometry_counterfactual_available: `{metrics.get('phase8_raw_overlap_geometry_counterfactual_available')}`",
        f"- phase8_raw_overlap_gate_pass: `{metrics.get('phase8_raw_overlap_gate_pass')}`",
        f"- phase8_bad_raw_overlap_median_improvement_vs_native: `{metrics.get('phase8_bad_raw_overlap_median_improvement_vs_native')}`",
        f"- phase8_good_raw_overlap_median_worsen_vs_native: `{metrics.get('phase8_good_raw_overlap_median_worsen_vs_native')}`",
        f"- runtime_action_allowed: `{final.get('runtime_action_allowed')}`",
        f"- ttt_allowed: `{final.get('ttt_allowed')}`",
        "",
        "Locked facts:",
        "",
        "1. v87 ended No-Go before runtime action.",
        "2. v87 high-observability geometry/local-shape scale proxy passed, but semantic_signal_pass=false.",
        "3. v87 Phase3 had no legal SUPPORT rows for C fit.",
        "4. v87 Phase4 no-refresh guard was unsafe because good_FPR=1.0.",
        "5. v87 Phase8 raw-overlap direct-pair counterfactual was available but failed the raw-overlap gate.",
        "6. v88 must test signed scale-mode consensus and native gauge-update mismatch, not rerun forbidden anchor/support/threshold families.",
    ]
    (args.out_dir / "v87_no_go_boundary.md").write_text("\n".join(boundary_lines) + "\n", encoding="utf-8")

    hypothesis_rows = [
        {
            "hypothesis_id": "H1",
            "claim": "v87 CONFLICT/STRESS is too coarse; signed local scale-mode consensus may be the useful signal.",
            "phase": "Phase1/2",
            "success_evidence": "mode/mismatch signal rho >=0.30, margin >=0.05, recall/FPR gate pass",
        },
        {
            "hypothesis_id": "H2",
            "claim": "good rows can contain local scale conflict if native update agrees with dominant mode.",
            "phase": "Phase3",
            "success_evidence": "MISMATCH_BAD recall >=0.60 and MISMATCH_GOOD FPR <=0.25",
        },
        {
            "hypothesis_id": "H3",
            "claim": "carrier is more likely merge/gauge boundary update than SWA source mask.",
            "phase": "Phase4/5",
            "success_evidence": "merge/gauge carrier and counterfactual beat random/shuffle while protecting good rows",
        },
        {
            "hypothesis_id": "H4",
            "claim": "SWA can only be tested as pairwise mode route, not source-only mask.",
            "phase": "Phase4/6",
            "success_evidence": "per-head/per-layer route lift or outlier excess beats controls without entropy collapse",
        },
        {
            "hypothesis_id": "H5",
            "claim": "semantic is a guard, not a main success claim unless it beats geometry and semantic-conf shuffle.",
            "phase": "Phase2/5",
            "success_evidence": "semantic-aware metric > geometry-only by >=0.05 and semantic-conf shuffle margin >=0.05",
        },
    ]
    write_csv(args.out_dir / "v88_hypothesis_matrix.csv", hypothesis_rows)

    evidence = {
        "phase": "Phase0_evidence_lock",
        "phase0_gate_pass": gate_pass,
        "checks": checks,
        "required_input_count": len(required_rows),
        "missing_required_inputs": [row for row in required_rows if not row["exists"]],
        "forbidden_repeat_count": len(FORBIDDEN_REPEATS),
        "raw_source_recovery": raw_check,
        "v87_boundary": {
            "final_status": final.get("final_status"),
            "blocker": final.get("blocker"),
            "phase2_highobs_gate_pass": metrics.get("phase2_highobs_gate_pass"),
            "phase2_S_overlap_rho": metrics.get("phase2_S_overlap_rho"),
            "phase2_semantic_signal_pass": metrics.get("phase2_semantic_signal_pass"),
            "phase3_valid_rows": metrics.get("phase3_valid_rows"),
            "phase3_blocker": metrics.get("phase3_blocker"),
            "phase4_good_FPR": metrics.get("phase4_good_FPR"),
            "phase8_raw_overlap_geometry_counterfactual_available": metrics.get("phase8_raw_overlap_geometry_counterfactual_available"),
            "phase8_raw_overlap_gate_pass": metrics.get("phase8_raw_overlap_gate_pass"),
            "phase8_bad_raw_overlap_median_improvement_vs_native": metrics.get("phase8_bad_raw_overlap_median_improvement_vs_native"),
            "runtime_action_allowed": final.get("runtime_action_allowed"),
            "ttt_allowed": final.get("ttt_allowed"),
        },
        "runtime_action_allowed": False,
        "ttt_allowed": False,
    }
    write_json(args.out_dir / "evidence_lock.json", evidence)
    write_json(args.out_dir / "phase0_gate_summary.json", evidence)

    print(f"phase0_gate_pass={gate_pass}")
    print(f"missing_required_inputs={len(evidence['missing_required_inputs'])}")
    print(f"selected_pair_count={raw_check['requested_pair_count']}")
    print(f"pairs_with_source_path={raw_check['pairs_with_source_path']}")
    print(f"raw_source_path_count={raw_check['raw_source_path_count']}")
    print(f"missing_raw_source_path_count={raw_check['missing_raw_source_path_count']}")
    print("runtime_action_allowed=False")
    print("ttt_allowed=False")


if __name__ == "__main__":
    main()
