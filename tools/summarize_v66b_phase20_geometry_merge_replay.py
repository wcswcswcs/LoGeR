#!/usr/bin/env python3
"""Summarize v66B Phase20B forced Sim(3) geometry merge replay."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

import numpy as np


ROOT = Path("results/kitti01_hmc_v2/acl2_v66b_dense_semantic_scale")
REPORT_ROOT = ROOT / "report_final"
DEFAULT_ROLLOUT_DIR = ROOT / "phase20_geometry_merge_replay" / "rollouts"
DEFAULT_PREFIX = "V66B_P20B_704_GEOM_MERGE"
H35_TRACE_DIR = (
    Path("results/kitti01_hmc_v2/acl2_v66b_artifacts/H35_FULL/rollouts")
    / "V66B_P19_704_GEOMETRY_TRACE_H35_NOOP"
)

REFERENCE_ATE = {
    "h35_noop_full": 43.650346,
    "phase14_best": 43.435810,
    "phase15_best": 43.462902,
    "phase17_best_semantic": 43.504096,
    "phase18_best": 43.496165,
}

RUN_SPECS = [
    {
        "run_suffix": "S1_GEOMETRY",
        "run_suffix_aliases": ["S1_GEOMETRY", "S1_GEOM"],
        "strategy": "S1_GEOMETRY_ONLY",
        "family": "geometry_control",
    },
    {
        "run_suffix": "S8_VERTICAL",
        "run_suffix_aliases": ["S8_VERTICAL", "S8_REAL"],
        "strategy": "S8_VERTICAL_STATIC_ONLY",
        "family": "semantic",
    },
    {
        "run_suffix": "S8_SHUFFLED",
        "run_suffix_aliases": ["S8_SHUFFLED"],
        "strategy": "S8_VERTICAL_STATIC_ONLY_SHUFFLED",
        "family": "shuffled_control",
    },
    {
        "run_suffix": "S11_SEMWEIGHT",
        "run_suffix_aliases": ["S11_SEMWEIGHT", "S11"],
        "strategy": "S11_SEMANTIC_GEOMETRY_WEIGHTED",
        "family": "semantic",
    },
]


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: List[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _float(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _count_lines(path: Path) -> Optional[int]:
    if not path.exists():
        return None
    with path.open("rb") as handle:
        return sum(1 for _ in handle)


def _read_ate(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {"exists": False}
    lines = [line.strip() for line in path.read_text(encoding="utf-8", errors="ignore").splitlines() if line.strip()]
    out: Dict[str, Any] = {"exists": True, "raw_lines": lines}
    for line in lines:
        if line.startswith("Average:"):
            parts = line.split()
            if len(parts) >= 3:
                out["ate_rmse"] = _float(parts[1])
                out["rot_rmse"] = _float(parts[2])
    return out


def _read_run_status(path: Path) -> List[str]:
    if not path.exists():
        return []
    return path.read_text(encoding="utf-8", errors="ignore").splitlines()[-4:]


def _trajectory_diff(path_a: Path, path_b: Path) -> Dict[str, Any]:
    if not path_a.exists() or not path_b.exists():
        return {"available": False, "a": str(path_a), "b": str(path_b)}
    arr_a = np.loadtxt(path_a)
    arr_b = np.loadtxt(path_b)
    if arr_a.shape != arr_b.shape:
        return {
            "available": True,
            "shape_match": False,
            "shape_a": list(arr_a.shape),
            "shape_b": list(arr_b.shape),
        }
    diff = np.abs(arr_a - arr_b)
    return {
        "available": True,
        "shape_match": True,
        "shape_a": list(arr_a.shape),
        "shape_b": list(arr_b.shape),
        "diff_lines": int(np.any(diff > 0.0, axis=1).sum()),
        "max_abs": float(diff.max()) if diff.size else 0.0,
        "mean_abs": float(diff.mean()) if diff.size else 0.0,
    }


def _read_merge_trace(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {"exists": False}
    rows = []
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    forced = [row for row in rows if bool(row.get("forced_merge_state_replay", False))]
    online = [row for row in rows if bool(row.get("online_semantic_merge_controller", False))]
    scale_source_rows = forced if forced else online
    scales = [
        _float(row.get("semantic_merge_scale", row.get("transform_scale_value")))
        for row in scale_source_rows
    ]
    scales = [value for value in scales if value is not None]
    loaded_counts = sorted({row.get("loaded_transform_count") for row in forced})
    residuals = [_float(row.get("semantic_merge_overlap_residual")) for row in online]
    residuals = [value for value in residuals if value is not None]
    return {
        "exists": True,
        "rows": len(rows),
        "forced_rows": len(forced),
        "online_rows": len(online),
        "loaded_transform_counts": loaded_counts,
        "scale_min": min(scales) if scales else None,
        "scale_mean": float(np.mean(scales)) if scales else None,
        "scale_max": max(scales) if scales else None,
        "semantic_overlap_residual_median": float(np.median(residuals)) if residuals else None,
        "semantic_overlap_residual_mean": float(np.mean(residuals)) if residuals else None,
        "first_forced_chunk": forced[0].get("chunk_idx") if forced else None,
        "last_forced_chunk": forced[-1].get("chunk_idx") if forced else None,
    }


def _input_trace_summary(strategy: str, trace_dir: Path) -> Dict[str, Any]:
    summary = _load_json(trace_dir / f"{strategy}.summary.json")
    keys = [
        "strategy",
        "row_count",
        "fit_failures",
        "median_overlap_residual",
        "scale_min",
        "scale_mean",
        "scale_max",
    ]
    return {key: summary.get(key) for key in keys if key in summary}


def _run_record(
    spec: Mapping[str, str],
    *,
    rollout_dir: Path,
    prefix: str,
    trace_dir: Path,
) -> Dict[str, Any]:
    suffixes = list(spec.get("run_suffix_aliases", [spec["run_suffix"]]))  # type: ignore[arg-type]
    run = f"{prefix}_{suffixes[0]}"
    run_dir = rollout_dir / run
    for suffix in suffixes:
        candidate = f"{prefix}_{suffix}"
        candidate_dir = rollout_dir / candidate
        if candidate_dir.exists():
            run = candidate
            run_dir = candidate_dir
            break
    ate = _read_ate(run_dir / "results_sim3/results_ate.txt")
    merge = _read_merge_trace(run_dir / "merge_state_trace.jsonl")
    diff_h35 = _trajectory_diff(H35_TRACE_DIR / "01.txt", run_dir / "01.txt")
    ate_rmse = ate.get("ate_rmse")
    merge_trace_complete = (
        merge.get("forced_rows") == 25
        or merge.get("online_rows") == 25
    )
    complete = (
        bool(ate.get("exists"))
        and ate_rmse is not None
        and _count_lines(run_dir / "hmc_state_hash.jsonl") == 25
        and merge_trace_complete
    )
    return {
        "run": run,
        "strategy": spec["strategy"],
        "family": spec["family"],
        "run_dir": str(run_dir),
        "complete": complete,
        "trajectory_rows": _count_lines(run_dir / "01.txt"),
        "hmc_state_rows": _count_lines(run_dir / "hmc_state_hash.jsonl"),
        "merge_trace_rows": merge.get("rows"),
        "forced_merge_rows": merge.get("forced_rows"),
        "online_semantic_merge_rows": merge.get("online_rows"),
        "loaded_transform_counts": json.dumps(merge.get("loaded_transform_counts", [])),
        "output_scale_min": merge.get("scale_min"),
        "output_scale_mean": merge.get("scale_mean"),
        "output_scale_max": merge.get("scale_max"),
        "semantic_overlap_residual_median": merge.get("semantic_overlap_residual_median"),
        "semantic_overlap_residual_mean": merge.get("semantic_overlap_residual_mean"),
        "ate_rmse": ate_rmse,
        "rot_rmse": ate.get("rot_rmse"),
        "ate_improvement_vs_h35": REFERENCE_ATE["h35_noop_full"] - ate_rmse
        if ate_rmse is not None
        else None,
        "ate_delta_vs_phase14_best": ate_rmse - REFERENCE_ATE["phase14_best"] if ate_rmse is not None else None,
        "ate_delta_vs_phase15_best": ate_rmse - REFERENCE_ATE["phase15_best"] if ate_rmse is not None else None,
        "ate_delta_vs_phase17_best_semantic": ate_rmse - REFERENCE_ATE["phase17_best_semantic"]
        if ate_rmse is not None
        else None,
        "ate_delta_vs_phase18_best": ate_rmse - REFERENCE_ATE["phase18_best"] if ate_rmse is not None else None,
        "diff_vs_h35_available": diff_h35.get("available"),
        "diff_vs_h35_shape_match": diff_h35.get("shape_match"),
        "diff_vs_h35_lines": diff_h35.get("diff_lines"),
        "diff_vs_h35_max_abs": diff_h35.get("max_abs"),
        "diff_vs_h35_mean_abs": diff_h35.get("mean_abs"),
        "input_trace": _input_trace_summary(spec["strategy"], trace_dir),
        "run_status_tail": _read_run_status(run_dir / "run_status.txt"),
        "ate_raw_lines": ate.get("raw_lines", []),
    }


def _best(rows: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    available = [row for row in rows if row.get("ate_rmse") is not None]
    return min(available, key=lambda row: row["ate_rmse"]) if available else None


def _delta_row(a: Dict[str, Any], b: Dict[str, Any], name: str) -> Dict[str, Any]:
    va = a.get("ate_rmse")
    vb = b.get("ate_rmse")
    return {
        "comparison": name,
        "a": a.get("run"),
        "b": b.get("run"),
        "ate_a": va,
        "ate_b": vb,
        "ate_delta_a_minus_b": (va - vb) if va is not None and vb is not None else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollout_dir", type=Path, default=DEFAULT_ROLLOUT_DIR)
    parser.add_argument("--report_dir", type=Path, default=REPORT_ROOT)
    parser.add_argument("--prefix", default=DEFAULT_PREFIX)
    parser.add_argument("--out_subdir", default="phase20_geometry_merge_replay")
    parser.add_argument("--top_key", default="phase20_geometry_merge_replay")
    parser.add_argument("--phase_tag", default="phase20b")
    parser.add_argument(
        "--trace_dir",
        type=Path,
        default=REPORT_ROOT / "phase20_geometry_merge_traces",
    )
    args = parser.parse_args()

    out_dir = args.report_dir / str(args.out_subdir)
    rows = [
        _run_record(spec, rollout_dir=args.rollout_dir, prefix=args.prefix, trace_dir=args.trace_dir)
        for spec in RUN_SPECS
    ]
    by_strategy = {row["strategy"]: row for row in rows}
    semantic_rows = [row for row in rows if row["family"] == "semantic"]
    control_rows = [row for row in rows if row["family"] != "semantic"]
    best_overall = _best(rows)
    best_semantic = _best(semantic_rows)
    best_control = _best(control_rows)

    deltas: List[Dict[str, Any]] = []
    if "S8_VERTICAL_STATIC_ONLY" in by_strategy and "S8_VERTICAL_STATIC_ONLY_SHUFFLED" in by_strategy:
        deltas.append(
            _delta_row(
                by_strategy["S8_VERTICAL_STATIC_ONLY"],
                by_strategy["S8_VERTICAL_STATIC_ONLY_SHUFFLED"],
                "S8_real_semantic_minus_shuffled_control",
            )
        )
    if "S8_VERTICAL_STATIC_ONLY" in by_strategy and "S1_GEOMETRY_ONLY" in by_strategy:
        deltas.append(
            _delta_row(
                by_strategy["S8_VERTICAL_STATIC_ONLY"],
                by_strategy["S1_GEOMETRY_ONLY"],
                "S8_real_semantic_minus_geometry_control",
            )
        )
    if "S11_SEMANTIC_GEOMETRY_WEIGHTED" in by_strategy and "S1_GEOMETRY_ONLY" in by_strategy:
        deltas.append(
            _delta_row(
                by_strategy["S11_SEMANTIC_GEOMETRY_WEIGHTED"],
                by_strategy["S1_GEOMETRY_ONLY"],
                "S11_semantic_weighted_minus_geometry_control",
            )
        )

    all_complete = all(bool(row.get("complete")) for row in rows)
    semantic_beats_best_control = (
        best_semantic is not None
        and best_control is not None
        and best_semantic.get("ate_rmse") is not None
        and best_control.get("ate_rmse") is not None
        and best_semantic["ate_rmse"] < best_control["ate_rmse"]
    )
    semantic_improvement = best_semantic.get("ate_improvement_vs_h35") if best_semantic else None
    no_go_reasons: List[str] = []
    if not all_complete:
        no_go_reasons.append("not_all_phase20b_runs_complete")
    if best_overall and best_overall.get("family") != "semantic":
        no_go_reasons.append("best_overall_run_is_control_not_semantic")
    if not semantic_beats_best_control:
        no_go_reasons.append("best_semantic_does_not_beat_best_control")
    if semantic_improvement is None or semantic_improvement < 0.5:
        no_go_reasons.append("best_semantic_improvement_vs_h35_below_0_5m_gate")
    if best_semantic and best_semantic.get("ate_delta_vs_phase14_best") is not None:
        if best_semantic["ate_delta_vs_phase14_best"] > 0.0:
            no_go_reasons.append("best_semantic_still_worse_than_phase14_best")

    phase_tag = str(args.phase_tag)
    online_mode = all(
        row.get("online_semantic_merge_rows") == 25 and row.get("forced_merge_rows") == 0
        for row in rows
    )
    gate_label = (
        "online semantic merge/gauge"
        if online_mode
        else "forced Sim(3) geometry merge replay"
    )
    status = f"{phase_tag}_incomplete"
    if all_complete:
        status_suffix = "online_semantic_merge" if online_mode else "geometry_merge_replay"
        status = f"pass_after_{phase_tag}_{status_suffix}" if not no_go_reasons else f"no_go_after_{phase_tag}_{status_suffix}"

    final_conclusion = (
        f"{phase_tag} {gate_label} is incomplete; no conclusion."
        if not all_complete
        else (
            f"{phase_tag} passes the {gate_label} gate."
            if not no_go_reasons
            else (
                f"No-Go after {phase_tag}. The {gate_label} path is executable and scale-bearing, but the "
                "valid 704F evidence does not show a semantic-specific repair that beats controls and prior best runs."
            )
        )
    )
    summary: Dict[str, Any] = {
        "status": status,
        "prefix": args.prefix,
        "rollout_dir": str(args.rollout_dir),
        "trace_dir": str(args.trace_dir),
        "h35_trace_dir": str(H35_TRACE_DIR),
        "reference_ate": REFERENCE_ATE,
        "all_complete": all_complete,
        "best_overall": best_overall,
        "best_semantic": best_semantic,
        "best_control": best_control,
        "semantic_beats_best_control": semantic_beats_best_control,
        "no_go_reasons": no_go_reasons,
        "deltas": deltas,
        "runs": rows,
        "final_conclusion": final_conclusion,
    }

    _write_json(out_dir / "phase20_geometry_merge_replay_summary.json", summary)
    _write_csv(out_dir / "phase20_geometry_merge_replay_by_run.csv", rows)
    _write_csv(out_dir / "phase20_geometry_merge_replay_deltas.csv", deltas)

    top_path = args.report_dir / "v66b_summary.json"
    top = _load_json(top_path)
    top["status"] = status
    top["final_conclusion"] = final_conclusion
    top[str(args.top_key)] = {
        "status": status,
        "summary_path": str(out_dir / "phase20_geometry_merge_replay_summary.json"),
        "by_run_csv": str(out_dir / "phase20_geometry_merge_replay_by_run.csv"),
        "deltas_csv": str(out_dir / "phase20_geometry_merge_replay_deltas.csv"),
        "all_complete": all_complete,
        "best_overall": best_overall,
        "best_semantic": best_semantic,
        "best_control": best_control,
        "no_go_reasons": no_go_reasons,
    }
    _write_json(top_path, top)

    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
