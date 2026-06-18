#!/usr/bin/env python3
"""Summarize v66B Phase18 merge/gauge load-same and donor probe."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np


RUNS = [
    "V66B_P17_704_SWA_OVERLAP_FULL_REPLACE_INTER_P025_SEM_A030_96F",
    "V66B_P17_704_SWA_OVERLAP_FULL_REPLACE_INTER_P025_RANDOM_A030_96F",
    "V66B_P18_704_MERGE_GAUGE_PROBE_SEM_LOADSAME",
    "V66B_P18_704_MERGE_GAUGE_PROBE_RANDOM_LOADSAME",
    "V66B_P18_704_MERGE_GAUGE_PROBE_SEM_USE_RANDOM_MERGE",
    "V66B_P18_704_MERGE_GAUGE_PROBE_RANDOM_USE_SEM_MERGE",
]

SEM_SRC = "V66B_P17_704_SWA_OVERLAP_FULL_REPLACE_INTER_P025_SEM_A030_96F"
RANDOM_SRC = "V66B_P17_704_SWA_OVERLAP_FULL_REPLACE_INTER_P025_RANDOM_A030_96F"
SEM_LOADSAME = "V66B_P18_704_MERGE_GAUGE_PROBE_SEM_LOADSAME"
RANDOM_LOADSAME = "V66B_P18_704_MERGE_GAUGE_PROBE_RANDOM_LOADSAME"
SEM_USE_RANDOM = "V66B_P18_704_MERGE_GAUGE_PROBE_SEM_USE_RANDOM_MERGE"
RANDOM_USE_SEM = "V66B_P18_704_MERGE_GAUGE_PROBE_RANDOM_USE_SEM_MERGE"

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


def _finite(value: float) -> float | None:
    return float(value) if math.isfinite(value) else None


def _write_csv(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    rows_list = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows_list:
        path.write_text("", encoding="utf-8")
        return
    fields: List[str] = []
    for row in rows_list:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows_list)


def _trajectory_diff(rollout_dir: Path, a: str, b: str) -> Dict[str, Any]:
    pa = rollout_dir / a / "01.txt"
    pb = rollout_dir / b / "01.txt"
    if not pa.is_file() or not pb.is_file():
        return {
            "comparison": f"{a}__vs__{b}",
            "a": a,
            "b": b,
            "available": False,
        }
    arr_a = np.loadtxt(pa)
    arr_b = np.loadtxt(pb)
    if arr_a.shape != arr_b.shape:
        return {
            "comparison": f"{a}__vs__{b}",
            "a": a,
            "b": b,
            "available": True,
            "shape_a": list(arr_a.shape),
            "shape_b": list(arr_b.shape),
            "shape_match": False,
        }
    diff = np.abs(arr_a - arr_b)
    return {
        "comparison": f"{a}__vs__{b}",
        "a": a,
        "b": b,
        "available": True,
        "shape_a": list(arr_a.shape),
        "shape_b": list(arr_b.shape),
        "shape_match": True,
        "diff_lines": int(np.any(diff > 0.0, axis=1).sum()),
        "max_abs": float(diff.max()) if diff.size else 0.0,
        "mean_abs": float(diff.mean()) if diff.size else 0.0,
    }


def _forced_count(path: Path) -> int | None:
    if not path.is_file():
        return None
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if bool(row.get("forced_merge_state_replay", False)):
                count += 1
    return count


def _run_row(
    run: str,
    *,
    phase9: Dict[str, Dict[str, str]],
    mech: Dict[str, Dict[str, str]],
    rollout_dir: Path,
    base_ate: float,
    phase14_ate: float,
    phase15_ate: float,
) -> Dict[str, Any]:
    p9 = phase9.get(run, {})
    mr = mech.get(run, {})
    artifact_dir = Path(str(p9.get("artifact_dir", rollout_dir / run)))
    traj = artifact_dir / "01.txt"
    merge_trace = artifact_dir / "merge_state_trace.jsonl"
    return {
        "run": run,
        "artifact_dir": str(artifact_dir),
        "frame_scope": p9.get("frame_scope", ""),
        "hmc_rows": int(_f(p9, "hmc_rows")) if math.isfinite(_f(p9, "hmc_rows")) else None,
        "trajectory_wc_rows": sum(1 for _ in traj.open("r", encoding="utf-8", errors="ignore"))
        if traj.is_file()
        else None,
        "merge_rows": sum(1 for _ in merge_trace.open("r", encoding="utf-8", errors="ignore"))
        if merge_trace.is_file()
        else None,
        "forced_merge_rows": _forced_count(merge_trace),
        "ate_rmse": _finite(_f(p9, "ate_rmse")),
        "ate_delta_vs_base": _finite(_f(p9, "ate_rmse") - base_ate),
        "ate_delta_vs_phase14_best": _finite(_f(p9, "ate_rmse") - phase14_ate),
        "ate_delta_vs_phase15_best": _finite(_f(p9, "ate_rmse") - phase15_ate),
        "rpe_trans": _finite(_f(p9, "rpe_trans")),
        "rpe_rot": _finite(_f(p9, "rpe_rot")),
        "traj_sha256": p9.get("traj_sha256", ""),
        "swa_replace_applied_total": _finite(_f(p9, "swa_overlap_source_replace_applied_total")),
        "swa_replace_alpha_avg": _finite(_f(p9, "swa_overlap_source_replace_alpha_avg")),
        "semantic_swa_role_control_mode": p9.get("semantic_swa_role_control_mode", ""),
        "semantic_swa_role_control_changed_fraction_avg": _finite(
            _f(p9, "semantic_swa_role_control_changed_fraction_avg")
        ),
        "local_sim3_mean": _finite(_f(mr, "local_sim3_chunk_ate_mean")),
        "intra_scale_variance_mean": _finite(_f(mr, "intra_scale_variance_mean")),
        "head_to_tail_ratio_mean": _finite(_f(mr, "head_to_tail_transfer_ratio_mean")),
        "future_after_overlap_mean": _finite(_f(mr, "future_after_overlap_error_pose_gt_proxy_mean")),
        "tail_after_overlap_mean": _finite(_f(mr, "tail_after_overlap_error_pose_gt_proxy_mean")),
    }


def _delta(rows: Dict[str, Dict[str, Any]], a: str, b: str, key: str) -> float | None:
    va = rows[a].get(key)
    vb = rows[b].get(key)
    if va is None or vb is None:
        return None
    return float(va - vb)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--report_dir",
        type=Path,
        default=Path("results/kitti01_hmc_v2/acl2_v66b_dense_semantic_scale/report_final"),
    )
    parser.add_argument(
        "--rollout_dir",
        type=Path,
        default=Path("results/kitti01_hmc_v2/acl2_v66b_dense_semantic_scale/phase9_parallel_continuation/rollouts"),
    )
    args = parser.parse_args()

    phase9 = _read_csv_by_run(args.report_dir / "phase9_parallel_continuation" / "phase9_parallel_metrics.csv")
    mech = _read_csv_by_run(args.report_dir / "phase10_mechanism_posthoc" / "rollout_mechanism_summary.csv")
    out_dir = args.report_dir / "phase18_merge_gauge_probe"

    base_ate = _f(phase9[BASE_RUN], "ate_rmse")
    phase14_ate = _f(phase9[PHASE14_BEST_RUN], "ate_rmse")
    phase15_ate = _f(phase9[PHASE15_BEST_RUN], "ate_rmse")
    rows = [
        _run_row(
            run,
            phase9=phase9,
            mech=mech,
            rollout_dir=args.rollout_dir,
            base_ate=base_ate,
            phase14_ate=phase14_ate,
            phase15_ate=phase15_ate,
        )
        for run in RUNS
    ]
    rows_by_run = {row["run"]: row for row in rows}

    trajectory_comparisons = [
        _trajectory_diff(args.rollout_dir, SEM_SRC, SEM_LOADSAME),
        _trajectory_diff(args.rollout_dir, RANDOM_SRC, RANDOM_LOADSAME),
        _trajectory_diff(args.rollout_dir, SEM_SRC, SEM_USE_RANDOM),
        _trajectory_diff(args.rollout_dir, RANDOM_SRC, RANDOM_USE_SEM),
        _trajectory_diff(args.rollout_dir, RANDOM_SRC, SEM_USE_RANDOM),
        _trajectory_diff(args.rollout_dir, SEM_SRC, RANDOM_USE_SEM),
    ]

    delta_rows: List[Dict[str, Any]] = []
    comparisons: List[Tuple[str, str, str]] = [
        ("sem_loadsame_vs_source", SEM_LOADSAME, SEM_SRC),
        ("random_loadsame_vs_source", RANDOM_LOADSAME, RANDOM_SRC),
        ("sem_use_random_vs_sem_source", SEM_USE_RANDOM, SEM_SRC),
        ("random_use_sem_vs_random_source", RANDOM_USE_SEM, RANDOM_SRC),
        ("sem_use_random_vs_random_source", SEM_USE_RANDOM, RANDOM_SRC),
        ("random_use_sem_vs_sem_source", RANDOM_USE_SEM, SEM_SRC),
    ]
    for name, a, b in comparisons:
        delta_rows.append(
            {
                "comparison": name,
                "a": a,
                "b": b,
                "ate_delta": _delta(rows_by_run, a, b, "ate_rmse"),
                "future_delta": _delta(rows_by_run, a, b, "future_after_overlap_mean"),
                "tail_delta": _delta(rows_by_run, a, b, "tail_after_overlap_mean"),
                "local_sim3_delta": _delta(rows_by_run, a, b, "local_sim3_mean"),
                "intra_scale_delta": _delta(rows_by_run, a, b, "intra_scale_variance_mean"),
            }
        )

    load_same_exact = all(
        c.get("shape_match")
        and c.get("diff_lines") == 0
        and c.get("max_abs") == 0.0
        for c in trajectory_comparisons[:2]
    )
    best_phase18 = min(rows[2:], key=lambda r: r["ate_rmse"])
    summary: Dict[str, Any] = {
        "status": "diagnostic_no_go_after_phase18_merge_gauge_probe",
        "base_run": BASE_RUN,
        "base_ate": base_ate,
        "phase14_best_run": PHASE14_BEST_RUN,
        "phase14_best_ate": phase14_ate,
        "phase15_best_run": PHASE15_BEST_RUN,
        "phase15_best_ate": phase15_ate,
        "run_count": 4,
        "all_phase18_runs_complete": all(
            r["hmc_rows"] == 25 and r["merge_rows"] == 25 and r["forced_merge_rows"] == 25 for r in rows[2:]
        ),
        "load_same_exact": load_same_exact,
        "best_phase18_run": best_phase18["run"],
        "best_phase18_ate": best_phase18["ate_rmse"],
        "best_phase18_delta_vs_phase14_best": best_phase18["ate_delta_vs_phase14_best"],
        "best_phase18_delta_vs_phase15_best": best_phase18["ate_delta_vs_phase15_best"],
        "conclusion": (
            "Forced merge/gauge replay is exact for load-same and is a real carrier, "
            "but simple semantic/random donor swaps do not produce a semantic-specific repair. "
            "Random using semantic merge improved ATE slightly, while semantic using random merge worsened."
        ),
        "runs": rows,
        "metric_deltas": delta_rows,
        "trajectory_comparisons": trajectory_comparisons,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / "phase18_merge_gauge_by_run.csv", rows)
    _write_csv(out_dir / "phase18_merge_gauge_metric_deltas.csv", delta_rows)
    _write_csv(out_dir / "phase18_merge_gauge_traj_diffs.csv", trajectory_comparisons)
    with (out_dir / "phase18_merge_gauge_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)

    print(json.dumps({
        "out_dir": str(out_dir.resolve()),
        "status": summary["status"],
        "load_same_exact": summary["load_same_exact"],
        "best_phase18_run": summary["best_phase18_run"],
        "best_phase18_ate": summary["best_phase18_ate"],
        "all_phase18_runs_complete": summary["all_phase18_runs_complete"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
