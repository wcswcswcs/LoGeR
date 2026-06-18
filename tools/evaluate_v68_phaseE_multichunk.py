#!/usr/bin/env python3
"""Aggregate ACL2 v68 Phase E READ multi-chunk trajectory gates."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.evaluate_v68_phaseD_read_smoke import (  # noqa: E402
    DEFAULT_GT,
    _best_control_value,
    _build_decision,
    _eval_run,
    _finite,
    _jsonable,
    _load_kitti_gt,
    _safe_ratio_improvement,
    _write_csv,
)


DEFAULT_CHUNKS = [6, 7, 8, 10, 12, 19, 20, 29, 30, 31, 32]
DEFAULT_ROOT = Path(
    "results/kitti01_hmc_v2/acl2_v68_integrated_cueconstruction/"
    "phaseE_read_multichunk/chunk32_v68_read_l5_7"
)
DEFAULT_RUNS = [
    "candidate",
    "geometry_only",
    "same_cue_random",
    "label_shuffled",
    "confidence_shuffled",
    "joint_shuffled",
    "native_no_read",
]
DEFAULT_CONTROLS = [
    "geometry_only",
    "same_cue_random",
    "label_shuffled",
    "confidence_shuffled",
    "joint_shuffled",
]
PHASEE_KEYS = [
    "head10_to_tail10_pose_sim3_rmse_m",
    "overlap3_to_future_pose_sim3_rmse_m",
]


def _parse_csv_ints(text: str) -> List[int]:
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def _write_decision_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            clean = {}
            for key in fieldnames:
                value = row.get(key)
                if isinstance(value, (dict, list, tuple)):
                    clean[key] = json.dumps(_jsonable(value), ensure_ascii=False, sort_keys=True)
                else:
                    clean[key] = value
            writer.writerow(clean)


def _metric_record(rows_by_name: Dict[str, Dict[str, Any]], *, key: str, candidate: str, baseline: str, controls: Sequence[str]) -> Dict[str, Any]:
    cand = rows_by_name.get(candidate, {})
    base = rows_by_name.get(baseline, {})
    cand_v = _finite(cand.get(key))
    base_v = _finite(base.get(key))
    best_ctrl = _best_control_value(rows_by_name, controls, key)
    ratio = _safe_ratio_improvement(base_v, cand_v)
    beats_controls = cand_v is not None and best_ctrl is not None and cand_v < best_ctrl
    return {
        "candidate": cand_v,
        "baseline": base_v,
        "best_control": best_ctrl,
        "candidate_minus_baseline": (cand_v - base_v) if cand_v is not None and base_v is not None else None,
        "candidate_minus_best_control": (cand_v - best_ctrl) if cand_v is not None and best_ctrl is not None else None,
        "improvement_vs_baseline_ratio": ratio,
        "beats_all_controls_for_key": beats_controls,
        "phaseE_chunk_key_pass": bool(beats_controls and ratio is not None and ratio > 0.0),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--gt", type=Path, default=DEFAULT_GT)
    parser.add_argument("--chunks", default=",".join(str(c) for c in DEFAULT_CHUNKS))
    parser.add_argument("--run", action="append", default=[], help="Run/case name under each chunk dir, repeatable")
    parser.add_argument("--candidate", default="candidate")
    parser.add_argument("--baseline", default="native_no_read")
    parser.add_argument("--control", action="append", default=[], help="Control case name, repeatable")
    parser.add_argument("--out-json", type=Path, default=None)
    parser.add_argument("--out-csv", type=Path, default=None)
    parser.add_argument("--out-rows-csv", type=Path, default=None)
    args = parser.parse_args()

    chunks = _parse_csv_ints(args.chunks)
    runs = args.run or DEFAULT_RUNS
    controls = args.control or DEFAULT_CONTROLS
    out_json = args.out_json or args.root / "phaseE_multichunk_summary.json"
    out_csv = args.out_csv or args.root / "phaseE_multichunk_decisions.csv"
    out_rows_csv = args.out_rows_csv or args.root / "phaseE_multichunk_run_metrics.csv"

    _, gt_poses_all, gt_pos_all = _load_kitti_gt(args.gt)
    all_rows: List[Dict[str, Any]] = []
    chunk_decisions: List[Dict[str, Any]] = []
    missing: List[str] = []

    for chunk in chunks:
        chunk_dir = args.root / f"chunk{chunk:02d}"
        rows: List[Dict[str, Any]] = []
        for run in runs:
            run_dir = chunk_dir / run
            traj = run_dir / "01.txt"
            if not traj.exists():
                missing.append(str(traj))
                continue
            row = _eval_run(run, run_dir, gt_poses_all, gt_pos_all)
            row["chunk"] = int(chunk)
            row["chunk_dir"] = str(chunk_dir)
            rows.append(row)
            all_rows.append(row)
        rows_by_name = {str(row["run"]): row for row in rows}
        phase_d_style = _build_decision(rows, candidate=args.candidate, baseline=args.baseline, controls=controls)
        metric_records = {
            key: _metric_record(
                rows_by_name,
                key=key,
                candidate=args.candidate,
                baseline=args.baseline,
                controls=controls,
            )
            for key in PHASEE_KEYS
        }
        chunk_decisions.append(
            {
                "chunk": int(chunk),
                "chunk_dir": str(chunk_dir),
                "phaseD_style_gate_pass": bool(phase_d_style.get("phaseD_gate_pass", False)),
                "phaseD_style_metric_passes": phase_d_style.get("metric_passes", []),
                "head_tail_beats_controls": bool(metric_records["head10_to_tail10_pose_sim3_rmse_m"]["beats_all_controls_for_key"]),
                "head_tail_improvement_vs_baseline_ratio": metric_records["head10_to_tail10_pose_sim3_rmse_m"]["improvement_vs_baseline_ratio"],
                "head_tail_phaseE_chunk_pass": bool(metric_records["head10_to_tail10_pose_sim3_rmse_m"]["phaseE_chunk_key_pass"]),
                "overlap_beats_controls": bool(metric_records["overlap3_to_future_pose_sim3_rmse_m"]["beats_all_controls_for_key"]),
                "overlap_improvement_vs_baseline_ratio": metric_records["overlap3_to_future_pose_sim3_rmse_m"]["improvement_vs_baseline_ratio"],
                "overlap_phaseE_chunk_pass": bool(metric_records["overlap3_to_future_pose_sim3_rmse_m"]["phaseE_chunk_key_pass"]),
                "metrics": metric_records,
            }
        )

    head_improvements = [
        float(row["head_tail_improvement_vs_baseline_ratio"])
        for row in chunk_decisions
        if row.get("head_tail_improvement_vs_baseline_ratio") is not None
    ]
    overlap_improvements = [
        float(row["overlap_improvement_vs_baseline_ratio"])
        for row in chunk_decisions
        if row.get("overlap_improvement_vs_baseline_ratio") is not None
    ]
    head_pass_count = int(sum(1 for row in chunk_decisions if row.get("head_tail_phaseE_chunk_pass")))
    overlap_pass_count = int(sum(1 for row in chunk_decisions if row.get("overlap_phaseE_chunk_pass")))
    head_median = float(statistics.median(head_improvements)) if head_improvements else None
    overlap_median = float(statistics.median(overlap_improvements)) if overlap_improvements else None
    phase_e_head_pass = bool(head_pass_count >= 4 and head_median is not None and head_median >= 0.05)
    phase_e_overlap_pass = bool(overlap_pass_count >= 4 and overlap_median is not None and overlap_median >= 0.05)

    summary = {
        "root": str(args.root),
        "gt": str(args.gt),
        "chunks": chunks,
        "candidate": args.candidate,
        "baseline": args.baseline,
        "controls": controls,
        "missing": missing,
        "head_tail_pass_count": head_pass_count,
        "overlap_pass_count": overlap_pass_count,
        "head_tail_median_improvement_vs_baseline_ratio": head_median,
        "overlap_median_improvement_vs_baseline_ratio": overlap_median,
        "phaseE_head_tail_pass": phase_e_head_pass,
        "phaseE_overlap_pass": phase_e_overlap_pass,
        "phaseE_gate_pass": bool(not missing and (phase_e_head_pass or phase_e_overlap_pass)),
        "rule": (
            "Phase E pass requires no missing target runs, >=4/11 chunks beating all controls "
            "on the same head_tail or overlap_to_future metric, and median improvement >=5% "
            "for that same metric."
        ),
        "chunk_decisions": chunk_decisions,
        "run_rows": all_rows,
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(_jsonable(summary), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_decision_csv(out_csv, chunk_decisions)
    _write_csv(out_rows_csv, all_rows)
    printable = {k: v for k, v in summary.items() if k not in {"run_rows", "chunk_decisions"}}
    print(json.dumps(_jsonable(printable), ensure_ascii=False, indent=2, sort_keys=True))
    print(f"wrote_json={out_json}")
    print(f"wrote_csv={out_csv}")
    print(f"wrote_rows_csv={out_rows_csv}")


if __name__ == "__main__":
    main()
