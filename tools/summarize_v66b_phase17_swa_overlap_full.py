#!/usr/bin/env python3
"""Summarize v66B Phase17 704F SWA overlap carrier validation."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List


PHASE17_RUNS = [
    "V66B_P17_704_SWA_OVERLAP_FULL_GATE_SOURCE_P050_SEM_RHO050_96F",
    "V66B_P17_704_SWA_OVERLAP_FULL_GATE_SOURCE_P050_RANDOM_RHO050_96F",
    "V66B_P17_704_SWA_OVERLAP_FULL_GATE_INTER_P025_SEM_RHO075_96F",
    "V66B_P17_704_SWA_OVERLAP_FULL_GATE_INTER_P025_RANDOM_RHO075_96F",
    "V66B_P17_704_SWA_OVERLAP_FULL_REPLACE_INTER_P025_SEM_A030_96F",
    "V66B_P17_704_SWA_OVERLAP_FULL_REPLACE_INTER_P025_RANDOM_A030_96F",
]

PAIRS = [
    (
        "gate_source_p050_rho050",
        "V66B_P17_704_SWA_OVERLAP_FULL_GATE_SOURCE_P050_SEM_RHO050_96F",
        "V66B_P17_704_SWA_OVERLAP_FULL_GATE_SOURCE_P050_RANDOM_RHO050_96F",
    ),
    (
        "gate_inter_p025_rho075",
        "V66B_P17_704_SWA_OVERLAP_FULL_GATE_INTER_P025_SEM_RHO075_96F",
        "V66B_P17_704_SWA_OVERLAP_FULL_GATE_INTER_P025_RANDOM_RHO075_96F",
    ),
    (
        "replace_inter_p025_a030",
        "V66B_P17_704_SWA_OVERLAP_FULL_REPLACE_INTER_P025_SEM_A030_96F",
        "V66B_P17_704_SWA_OVERLAP_FULL_REPLACE_INTER_P025_RANDOM_A030_96F",
    ),
]

BASE_RUN = "V66B_P9_704_TTT_BASE_DENSE_IGNORE_SELECTOR_RULE"
PHASE14_BEST_RUN = "V66B_P9_704_TTT_SELECTOR_RULE_SOFT_SEM_96F"
PHASE15_BEST_RUN = "V66B_P15_704_TTT_CAUSAL_SCALE_BROADCAST_MID130_SEM"


def _read_csv_by_run(path: Path) -> Dict[str, Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return {str(row.get("run", "")): row for row in csv.DictReader(handle)}


def _f(row: Dict[str, Any], key: str) -> float:
    value = row.get(key)
    if value in (None, ""):
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _finite_or_none(value: float) -> float | None:
    return float(value) if math.isfinite(value) else None


def _improvement(base: float, value: float) -> float | None:
    if not math.isfinite(base) or not math.isfinite(value) or abs(base) < 1e-12:
        return None
    return float((base - value) / abs(base))


def _write_csv(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    rows_list = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows_list:
        path.write_text("", encoding="utf-8")
        return
    fields: List[str] = []
    for row in rows_list:
        for key in row.keys():
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows_list)


def _run_row(
    *,
    run: str,
    phase9: Dict[str, Dict[str, str]],
    mech: Dict[str, Dict[str, str]],
    base_mech: Dict[str, str],
    base_ate: float,
    phase14_ate: float,
    phase15_ate: float,
) -> Dict[str, Any]:
    p9 = phase9.get(run, {})
    mr = mech.get(run, {})
    out: Dict[str, Any] = {
        "run": run,
        "artifact_dir": p9.get("artifact_dir", ""),
        "frame_scope": p9.get("frame_scope", ""),
        "hmc_rows": int(_f(p9, "hmc_rows")) if math.isfinite(_f(p9, "hmc_rows")) else None,
        "num_chunks": int(_f(p9, "num_chunks")) if math.isfinite(_f(p9, "num_chunks")) else None,
        "trajectory_rows": None,
        "ate_rmse": _finite_or_none(_f(p9, "ate_rmse")),
        "ate_delta_vs_base": _finite_or_none(_f(p9, "ate_rmse") - base_ate),
        "ate_delta_vs_phase14_best": _finite_or_none(_f(p9, "ate_rmse") - phase14_ate),
        "ate_delta_vs_phase15_best": _finite_or_none(_f(p9, "ate_rmse") - phase15_ate),
        "rpe_trans": _finite_or_none(_f(p9, "rpe_trans")),
        "rpe_rot": _finite_or_none(_f(p9, "rpe_rot")),
        "traj_sha256": p9.get("traj_sha256", ""),
        "semantic_role_consumed_rows": int(_f(p9, "semantic_role_consumed_rows"))
        if math.isfinite(_f(p9, "semantic_role_consumed_rows"))
        else None,
        "semantic_swa_role_control_mode": p9.get("semantic_swa_role_control_mode", ""),
        "semantic_swa_role_control_applied_rows": int(_f(p9, "semantic_swa_role_control_applied_rows"))
        if math.isfinite(_f(p9, "semantic_swa_role_control_applied_rows"))
        else None,
        "semantic_swa_role_control_changed_fraction_avg": _finite_or_none(
            _f(p9, "semantic_swa_role_control_changed_fraction_avg")
        ),
        "semantic_role_swa_negative_scale": _finite_or_none(_f(p9, "semantic_role_swa_negative_scale")),
        "semantic_role_swa_protect_scale": _finite_or_none(_f(p9, "semantic_role_swa_protect_scale")),
        "semantic_role_swa_protect_adjusted_rows": int(_f(p9, "semantic_role_swa_protect_adjusted_rows"))
        if math.isfinite(_f(p9, "semantic_role_swa_protect_adjusted_rows"))
        else None,
        "swa_gate_applied_total": _finite_or_none(_f(p9, "swa_overlap_source_gate_applied_total")),
        "swa_gate_delta_avg": _finite_or_none(_f(p9, "swa_overlap_source_gate_delta_avg")),
        "swa_gate_score_mean_avg": _finite_or_none(_f(p9, "swa_overlap_source_gate_score_mean_avg")),
        "swa_replace_applied_total": _finite_or_none(_f(p9, "swa_overlap_source_replace_applied_total")),
        "swa_replace_alpha_avg": _finite_or_none(_f(p9, "swa_overlap_source_replace_alpha_avg")),
        "swa_replace_score_mean_avg": _finite_or_none(_f(p9, "swa_overlap_source_replace_score_mean_avg")),
        "local_sim3_mean": _finite_or_none(_f(mr, "local_sim3_chunk_ate_mean")),
        "intra_scale_variance_mean": _finite_or_none(_f(mr, "intra_scale_variance_mean")),
        "head_to_tail_ratio_mean": _finite_or_none(_f(mr, "head_to_tail_transfer_ratio_mean")),
        "future_after_overlap_mean": _finite_or_none(_f(mr, "future_after_overlap_error_pose_gt_proxy_mean")),
        "tail_after_overlap_mean": _finite_or_none(_f(mr, "tail_after_overlap_error_pose_gt_proxy_mean")),
    }
    artifact_dir = Path(str(out["artifact_dir"]))
    traj = artifact_dir / "01.txt"
    if traj.is_file():
        out["trajectory_rows"] = sum(1 for _ in traj.open("r", encoding="utf-8", errors="ignore"))

    for key, out_key in [
        ("local_sim3_chunk_ate_mean", "local_sim3_improvement_vs_base"),
        ("intra_scale_variance_mean", "intra_scale_improvement_vs_base"),
        ("head_to_tail_transfer_ratio_mean", "head_to_tail_improvement_vs_base"),
        ("future_after_overlap_error_pose_gt_proxy_mean", "future_after_overlap_improvement_vs_base"),
        ("tail_after_overlap_error_pose_gt_proxy_mean", "tail_after_overlap_improvement_vs_base"),
    ]:
        out[out_key] = _improvement(_f(base_mech, key), _f(mr, key))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--report_dir",
        type=Path,
        default=Path("results/kitti01_hmc_v2/acl2_v66b_dense_semantic_scale/report_final"),
    )
    args = parser.parse_args()

    phase9_csv = args.report_dir / "phase9_parallel_continuation" / "phase9_parallel_metrics.csv"
    mech_csv = args.report_dir / "phase10_mechanism_posthoc" / "rollout_mechanism_summary.csv"
    out_dir = args.report_dir / "phase17_swa_overlap_full"
    phase9 = _read_csv_by_run(phase9_csv)
    mech = _read_csv_by_run(mech_csv)

    base_ate = _f(phase9[BASE_RUN], "ate_rmse")
    phase14_ate = _f(phase9[PHASE14_BEST_RUN], "ate_rmse")
    phase15_ate = _f(phase9[PHASE15_BEST_RUN], "ate_rmse")
    base_mech = mech[BASE_RUN]

    rows = [
        _run_row(
            run=run,
            phase9=phase9,
            mech=mech,
            base_mech=base_mech,
            base_ate=base_ate,
            phase14_ate=phase14_ate,
            phase15_ate=phase15_ate,
        )
        for run in PHASE17_RUNS
    ]

    by_run = {row["run"]: row for row in rows}
    pair_rows: List[Dict[str, Any]] = []
    for pair_name, sem_run, random_run in PAIRS:
        sem = by_run[sem_run]
        rnd = by_run[random_run]
        pair_rows.append(
            {
                "pair": pair_name,
                "semantic_run": sem_run,
                "random_run": random_run,
                "semantic_ate": sem["ate_rmse"],
                "random_ate": rnd["ate_rmse"],
                "semantic_minus_random_ate": sem["ate_rmse"] - rnd["ate_rmse"],
                "semantic_future_improvement_vs_base": sem["future_after_overlap_improvement_vs_base"],
                "random_future_improvement_vs_base": rnd["future_after_overlap_improvement_vs_base"],
                "semantic_tail_improvement_vs_base": sem["tail_after_overlap_improvement_vs_base"],
                "random_tail_improvement_vs_base": rnd["tail_after_overlap_improvement_vs_base"],
                "semantic_minus_random_future_error": sem["future_after_overlap_mean"]
                - rnd["future_after_overlap_mean"],
                "semantic_minus_random_tail_error": sem["tail_after_overlap_mean"] - rnd["tail_after_overlap_mean"],
                "semantic_swa_action_rows": sem["swa_gate_applied_total"] or sem["swa_replace_applied_total"],
                "random_swa_action_rows": rnd["swa_gate_applied_total"] or rnd["swa_replace_applied_total"],
            }
        )

    semantic_rows = [r for r in rows if r["semantic_swa_role_control_mode"] == "none"]
    best_semantic = min(semantic_rows, key=lambda r: r["ate_rmse"])
    best_any = min(rows, key=lambda r: r["ate_rmse"])
    gate_pass_semantic = [
        r["run"]
        for r in semantic_rows
        if (r["future_after_overlap_improvement_vs_base"] or 0.0) >= 0.10
        and (r["tail_after_overlap_improvement_vs_base"] or 0.0) >= 0.10
        and r["ate_delta_vs_base"] <= 0.0
    ]

    summary: Dict[str, Any] = {
        "status": "no_go_after_phase17_swa_overlap_full",
        "audit_note": (
            "Run names end with _96F because the reused launcher defaulted an empty RUN_SUFFIX "
            "through ${RUN_SUFFIX:-_96F}; frame_scope/trajectory_rows confirm these are 704F runs."
        ),
        "base_run": BASE_RUN,
        "base_ate": base_ate,
        "phase14_best_run": PHASE14_BEST_RUN,
        "phase14_best_ate": phase14_ate,
        "phase15_best_run": PHASE15_BEST_RUN,
        "phase15_best_ate": phase15_ate,
        "run_count": len(rows),
        "all_runs_complete": all(r["hmc_rows"] == 25 and r["trajectory_rows"] == 705 for r in rows),
        "best_semantic_run": best_semantic["run"],
        "best_semantic_ate": best_semantic["ate_rmse"],
        "best_semantic_delta_vs_base": best_semantic["ate_delta_vs_base"],
        "best_semantic_delta_vs_phase14_best": best_semantic["ate_delta_vs_phase14_best"],
        "best_semantic_delta_vs_phase15_best": best_semantic["ate_delta_vs_phase15_best"],
        "best_any_run": best_any["run"],
        "best_any_ate": best_any["ate_rmse"],
        "semantic_gate_pass_runs": gate_pass_semantic,
        "conclusion": (
            "SWA overlap carrier remains active at 704F, but semantic runs do not satisfy "
            "future/tail >=10% improvement with ATE non-regression, and best Phase17 remains "
            "worse than Phase14/Phase15 best ATE."
        ),
        "runs": rows,
        "pairs": pair_rows,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / "phase17_swa_overlap_full_by_run.csv", rows)
    _write_csv(out_dir / "phase17_swa_overlap_full_pairs.csv", pair_rows)
    with (out_dir / "phase17_swa_overlap_full_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)

    print(json.dumps({
        "out_dir": str(out_dir.resolve()),
        "status": summary["status"],
        "best_semantic_run": summary["best_semantic_run"],
        "best_semantic_ate": summary["best_semantic_ate"],
        "best_any_run": summary["best_any_run"],
        "best_any_ate": summary["best_any_ate"],
        "all_runs_complete": summary["all_runs_complete"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
