#!/usr/bin/env python3
"""Diagnose v80 semantic false positives in boundary merge/gauge controls.

This is a read-only artifact audit. It compares PhaseE metrics against
merge_state_trace fields for one already-materialized root, focusing on whether
semantic qscale/scale-state decisions explain why a semantic candidate loses to
geometry-only or randomized controls.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


REPORT_ROOT = Path(
    "results/kitti01_hmc_v2/acl2_v80tf_multiseq_goodbad_semantic_three_memory_control/report_final"
)
DEFAULT_ROOT = REPORT_ROOT / "phase9_seq01_thingstuff_radio_qscale_boundary09_alpha160_qscale_weighted_gate_chunks009"
DEFAULT_METRICS = DEFAULT_ROOT / "thingstuff_radio_qscale_boundary09_alpha160_qscale_weighted_gate_run_metrics.csv"
DEFAULT_DECISIONS = DEFAULT_ROOT / "thingstuff_radio_qscale_boundary09_alpha160_qscale_weighted_gate_decisions.csv"
DEFAULT_OUT = REPORT_ROOT / "phase9_seq01_boundary09_semantic_false_positive_merge_gauge_audit"
DEFAULT_RUNS = (
    "native_no_swa",
    "thingstuff_radio_qscale",
    "thingstuff_radio_qscale_random",
    "thingstuff_radio_qscale_shuffled",
    "geometry_only",
)
METRIC_KEYS = (
    "head10_to_tail10_pose_sim3_rmse_m",
    "overlap3_to_future_pose_sim3_rmse_m",
    "local_sim3_ate_rmse_m",
    "scale_cv_head_mid_tail_pose_sim3",
)
TRACE_KEYS = (
    "semantic_merge_strategy",
    "semantic_merge_control_type",
    "semantic_merge_scale",
    "semantic_merge_candidate_scale",
    "semantic_merge_blend_scale",
    "semantic_merge_qscale_observability",
    "semantic_merge_radio_handoff_qscale_observability",
    "semantic_merge_radio_handoff_weighted_qscale_observability",
    "semantic_merge_radio_handoff_weighted_stable_mean",
    "semantic_merge_radio_handoff_weighted_risk_mean",
    "semantic_merge_native_overlap_guard_rejected",
    "online_scale_state_gate_enabled",
    "online_scale_state_gate_policy",
    "online_scale_state_gate_pass",
    "online_scale_state_gate_reason",
    "online_scale_state_gate_qscale",
    "online_scale_state_gate_strategy_core",
    "online_scale_state_gate_control_type",
    "online_scale_state_active",
    "online_scale_state_action",
    "online_scale_state_input_scale",
    "online_scale_state_output_scale",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--metrics-csv", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--decisions-csv", type=Path, default=DEFAULT_DECISIONS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--candidate", default="thingstuff_radio_qscale")
    parser.add_argument("--baseline", default="native_no_swa")
    parser.add_argument(
        "--controls",
        default="thingstuff_radio_qscale_random,thingstuff_radio_qscale_shuffled,geometry_only",
    )
    parser.add_argument("--runs", default=",".join(DEFAULT_RUNS))
    return parser.parse_args()


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _safe_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in fieldnames})


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(_jsonable(value), ensure_ascii=False, sort_keys=True)
    return value


def _selected_trace(root: Path, run: str) -> dict[str, Any]:
    rows = _read_jsonl(root / "chunk09" / run / "merge_state_trace.jsonl")
    selected = [row for row in rows if int(row.get("local_chunk_idx", -1)) == 1]
    return selected[-1] if selected else (rows[-1] if rows else {})


def _rank_runs(rows: list[dict[str, Any]], metric: str) -> list[str]:
    finite = [row for row in rows if _safe_float(row.get(metric)) is not None]
    return [str(row["run"]) for row in sorted(finite, key=lambda row: float(row[metric]))]


def _positions(path: Path) -> list[tuple[float, float, float]]:
    if not path.exists():
        return []
    out: list[tuple[float, float, float]] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = [float(part) for part in line.split()]
            if len(parts) >= 8:
                out.append((parts[1], parts[2], parts[3]))
            elif len(parts) == 12:
                out.append((parts[3], parts[7], parts[11]))
    return out


def _norm3(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return math.sqrt(sum((float(a[i]) - float(b[i])) ** 2 for i in range(3)))


def _median(values: list[float]) -> float | None:
    values = sorted(value for value in values if math.isfinite(value))
    if not values:
        return None
    mid = len(values) // 2
    if len(values) % 2:
        return float(values[mid])
    return float((values[mid - 1] + values[mid]) / 2.0)


def _trajectory_features(run_dir: Path) -> dict[str, Any]:
    pos = _positions(run_dir / "01.txt")
    steps = [_norm3(pos[i + 1], pos[i]) for i in range(len(pos) - 1)]
    boundary_idx = 28 if len(steps) > 28 else max(len(steps) // 2, 0)
    pre = steps[max(0, boundary_idx - 10) : boundary_idx]
    post = steps[boundary_idx + 1 : min(len(steps), boundary_idx + 11)]
    pre_med = _median(pre)
    post_med = _median(post)
    boundary = steps[boundary_idx] if boundary_idx < len(steps) else None
    return {
        "traj_step_count": len(steps),
        "traj_boundary_idx": boundary_idx,
        "traj_median_step": _median(steps),
        "traj_pre_boundary_median_step": pre_med,
        "traj_boundary_step": boundary,
        "traj_post_boundary_median_step": post_med,
        "traj_boundary_over_pre": (
            float(boundary) / float(pre_med)
            if boundary is not None and pre_med is not None and abs(float(pre_med)) > 1.0e-12
            else None
        ),
        "traj_post_over_pre": (
            float(post_med) / float(pre_med)
            if post_med is not None and pre_med is not None and abs(float(pre_med)) > 1.0e-12
            else None
        ),
    }


def _fmt_float(value: Any, digits: int = 6) -> str:
    out = _safe_float(value)
    if out is None:
        return "NA"
    return f"{out:.{digits}f}"


def main() -> None:
    args = _parse_args()
    runs = [part.strip() for part in args.runs.split(",") if part.strip()]
    controls = [part.strip() for part in args.controls.split(",") if part.strip()]
    metric_rows_raw = _read_csv(args.metrics_csv)
    metric_by_run = {str(row.get("run")): row for row in metric_rows_raw}

    rows: list[dict[str, Any]] = []
    for run in runs:
        metric = dict(metric_by_run.get(run, {}))
        trace = _selected_trace(args.root, run)
        row: dict[str, Any] = {
            "run": run,
            "is_candidate": run == args.candidate,
            "is_baseline": run == args.baseline,
            "is_control": run in controls,
            "run_dir": str(args.root / "chunk09" / run),
        }
        row.update(_trajectory_features(args.root / "chunk09" / run))
        for key in METRIC_KEYS:
            row[key] = _safe_float(metric.get(key))
        for key in TRACE_KEYS:
            row[key] = trace.get(key)
        rows.append(row)

    candidate = next((row for row in rows if row["run"] == args.candidate), {})
    baseline = next((row for row in rows if row["run"] == args.baseline), {})
    control_rows = [row for row in rows if row["run"] in controls]

    metric_summary: dict[str, Any] = {}
    for metric in METRIC_KEYS:
        ranked = _rank_runs(rows, metric)
        cand_value = _safe_float(candidate.get(metric))
        base_value = _safe_float(baseline.get(metric))
        finite_controls = [row for row in control_rows if _safe_float(row.get(metric)) is not None]
        best_control = min(finite_controls, key=lambda row: float(row[metric])) if finite_controls else {}
        best_control_value = _safe_float(best_control.get(metric)) if best_control else None
        metric_summary[metric] = {
            "ranked_runs_low_is_best": ranked,
            "candidate": cand_value,
            "baseline": base_value,
            "best_control_run": best_control.get("run"),
            "best_control": best_control_value,
            "candidate_minus_baseline": (
                cand_value - base_value if cand_value is not None and base_value is not None else None
            ),
            "candidate_minus_best_control": (
                cand_value - best_control_value
                if cand_value is not None and best_control_value is not None
                else None
            ),
            "candidate_beats_all_controls": bool(
                cand_value is not None
                and all(
                    _safe_float(row.get(metric)) is not None and cand_value < float(row[metric])
                    for row in finite_controls
                )
            ),
        }

    cand_wq = _safe_float(candidate.get("semantic_merge_radio_handoff_weighted_qscale_observability"))
    random_row = next((row for row in rows if row["run"] == "thingstuff_radio_qscale_random"), {})
    random_wq = _safe_float(random_row.get("semantic_merge_radio_handoff_weighted_qscale_observability"))
    shuffled_row = next((row for row in rows if row["run"] == "thingstuff_radio_qscale_shuffled"), {})
    geometry_row = next((row for row in rows if row["run"] == "geometry_only"), {})

    qscale_gap = abs(cand_wq - random_wq) if cand_wq is not None and random_wq is not None else None
    candidate_clamped = bool(candidate.get("online_scale_state_active"))
    random_clamped = bool(random_row.get("online_scale_state_active"))
    geometry_better_on_core = all(
        (
            _safe_float(geometry_row.get(metric)) is not None
            and _safe_float(candidate.get(metric)) is not None
            and float(geometry_row[metric]) < float(candidate[metric])
        )
        for metric in ("head10_to_tail10_pose_sim3_rmse_m", "overlap3_to_future_pose_sim3_rmse_m")
    )
    shuffled_better_on_core = all(
        (
            _safe_float(shuffled_row.get(metric)) is not None
            and _safe_float(candidate.get(metric)) is not None
            and float(shuffled_row[metric]) < float(candidate[metric])
        )
        for metric in ("head10_to_tail10_pose_sim3_rmse_m", "overlap3_to_future_pose_sim3_rmse_m")
    )
    candidate_post_over_pre = _safe_float(candidate.get("traj_post_over_pre"))
    geometry_post_over_pre = _safe_float(geometry_row.get("traj_post_over_pre"))
    random_post_over_pre = _safe_float(random_row.get("traj_post_over_pre"))

    conclusion = {
        "semantic_false_positive_confirmed": bool(
            candidate_clamped
            and random_clamped
            and qscale_gap is not None
            and qscale_gap < 0.02
            and geometry_better_on_core
        ),
        "reason": (
            "candidate and random have near-identical weighted qscale and both clamp, while "
            "geometry-only and shuffled no-clamp controls achieve lower head-tail/overlap errors"
        ),
        "candidate_weighted_qscale": cand_wq,
        "random_weighted_qscale": random_wq,
        "candidate_random_weighted_qscale_abs_gap": qscale_gap,
        "candidate_clamped": candidate_clamped,
        "random_clamped": random_clamped,
        "geometry_better_on_head_overlap": geometry_better_on_core,
        "shuffled_better_on_head_overlap": shuffled_better_on_core,
        "candidate_traj_post_over_pre": candidate_post_over_pre,
        "random_traj_post_over_pre": random_post_over_pre,
        "geometry_traj_post_over_pre": geometry_post_over_pre,
        "geometry_post_over_pre_above_candidate": bool(
            candidate_post_over_pre is not None
            and geometry_post_over_pre is not None
            and geometry_post_over_pre > candidate_post_over_pre
        ),
        "v80_goal_achieved": False,
        "method_gate_claimed": False,
        "runtime_promotion_allowed": False,
        "next_action": (
            "Stop qscale threshold tuning on chunk009. Add geometry residual direction, boundary step, "
            "and tail/future proxy terms to the merge/gauge controller before any canary promotion."
        ),
    }

    summary = {
        "schema": "acl2_v80_semantic_false_positive_merge_gauge_audit_v1",
        "status": "diagnostic_semantic_false_positive_audit",
        "root": str(args.root),
        "metrics_csv": str(args.metrics_csv),
        "decisions_csv": str(args.decisions_csv),
        "candidate": args.candidate,
        "baseline": args.baseline,
        "controls": controls,
        "metric_summary": metric_summary,
        "conclusion": conclusion,
        "rows": rows,
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.out_dir / "semantic_false_positive_rows.csv", rows)
    _write_json(args.out_dir / "semantic_false_positive_summary.json", summary)

    report_lines = [
        "# ACL2 v80 boundary09 semantic false-positive merge/gauge audit",
        "",
        "## Decision",
        "",
        f"- semantic_false_positive_confirmed: `{conclusion['semantic_false_positive_confirmed']}`",
        f"- candidate_weighted_qscale: `{_fmt_float(cand_wq)}`",
        f"- random_weighted_qscale: `{_fmt_float(random_wq)}`",
        f"- candidate_random_weighted_qscale_abs_gap: `{_fmt_float(qscale_gap)}`",
        f"- candidate_traj_post_over_pre: `{_fmt_float(candidate_post_over_pre)}`",
        f"- geometry_traj_post_over_pre: `{_fmt_float(geometry_post_over_pre)}`",
        f"- v80_goal_achieved: `{conclusion['v80_goal_achieved']}`",
        f"- method_gate_claimed: `{conclusion['method_gate_claimed']}`",
        "",
        "## Metric Ranking",
        "",
    ]
    for metric, payload in metric_summary.items():
        report_lines.append(f"- {metric}: `{payload['ranked_runs_low_is_best']}`")
    report_lines.extend(
        [
            "",
            "## Interpretation",
            "",
            conclusion["reason"],
            "",
            "The current qscale proxy is not selective enough for boundary09: it marks the semantic candidate and same-mass random as similarly reliable, while the no-clamp geometry-only and shuffled controls have better downstream geometry.",
            "",
            "## Next Action",
            "",
            conclusion["next_action"],
            "",
        ]
    )
    (args.out_dir / "semantic_false_positive_report.md").write_text("\n".join(report_lines), encoding="utf-8")
    print(json.dumps(_jsonable({k: v for k, v in summary.items() if k != "rows"}), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
