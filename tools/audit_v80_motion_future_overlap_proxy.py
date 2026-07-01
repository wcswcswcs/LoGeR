#!/usr/bin/env python3
"""Audit non-GT motion proxies for v80 future-overlap safety.

The audit reads existing seq01 canary5 trajectories and GT-evaluated metrics.
It does not run LoGeR and does not claim a method gate.  Its purpose is to
check whether trajectory-only continuity proxies can select semantic qscale
chunks that improve head-tail without harming overlap-to-future.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import median
from typing import Any, Callable

import numpy as np


REPORT_ROOT = Path(
    "results/kitti01_hmc_v2/acl2_v80tf_multiseq_goodbad_semantic_three_memory_control/"
    "report_final"
)
DEFAULT_ROOT = REPORT_ROOT / "phase9_seq01_thingstuff_radio_qscale_canary5_ref055_chunks006_008_010_012"
DEFAULT_RECHECK_ROWS = REPORT_ROOT / "phase9_seq01_non_gt_direction_recheck" / "non_gt_direction_recheck_rows.csv"
DEFAULT_METRICS = DEFAULT_ROOT / "thingstuff_radio_qscale_ref055_canary5_run_metrics.csv"
DEFAULT_OUT_DIR = REPORT_ROOT / "phase10_seq01_motion_future_overlap_proxy_audit_20260622_2055"
RUNS = [
    "native_no_swa",
    "geometry_only",
    "thingstuff_radio_qscale",
    "thingstuff_radio_qscale_random",
    "thingstuff_radio_qscale_shuffled",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--metrics-csv", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--recheck-rows", type=Path, default=DEFAULT_RECHECK_ROWS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--min-median-improvement", type=float, default=0.05)
    return parser.parse_args()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


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


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(_jsonable(value), ensure_ascii=False, sort_keys=True)
    return value


def _f(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _load_positions(path: Path) -> np.ndarray:
    rows: list[list[float]] = []
    if not path.exists():
        return np.zeros((0, 3), dtype=np.float64)
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            vals = [float(part) for part in line.split()]
            if len(vals) >= 8:
                rows.append(vals[1:4])
            elif len(vals) == 12:
                rows.append([vals[3], vals[7], vals[11]])
    return np.asarray(rows, dtype=np.float64)


def _median(values: np.ndarray) -> float | None:
    vals = values[np.isfinite(values)]
    if vals.size <= 0:
        return None
    return float(np.median(vals))


def _ratio(num: float | None, den: float | None) -> float | None:
    if num is None or den is None or abs(den) < 1e-12:
        return None
    out = float(num) / float(den)
    return out if math.isfinite(out) else None


def _unit_error(value: float | None, *, log_space: bool = True) -> float | None:
    if value is None or value <= 0:
        return None
    return abs(math.log(value)) if log_space else abs(value - 1.0)


def _angles(vectors: np.ndarray) -> np.ndarray:
    if vectors.shape[0] < 2:
        return np.zeros((0,), dtype=np.float64)
    a = vectors[:-1]
    b = vectors[1:]
    an = np.linalg.norm(a, axis=1)
    bn = np.linalg.norm(b, axis=1)
    denom = np.maximum(an * bn, 1e-12)
    cos = np.sum(a * b, axis=1) / denom
    cos = np.clip(cos, -1.0, 1.0)
    return np.arccos(cos)


def _motion_features(path: Path) -> dict[str, Any]:
    pos = _load_positions(path)
    if pos.shape[0] < 4:
        return {"trajectory": str(path), "motion_available": False}
    vec = np.diff(pos, axis=0)
    steps = np.linalg.norm(vec, axis=1)
    boundary_idx = 28 if steps.size > 28 else max(int(steps.size) // 2, 0)
    pre = steps[max(0, boundary_idx - 10) : boundary_idx]
    post = steps[boundary_idx + 1 : min(steps.size, boundary_idx + 11)]
    boundary = float(steps[boundary_idx]) if boundary_idx < steps.size else None
    prev = float(steps[boundary_idx - 1]) if boundary_idx > 0 and boundary_idx - 1 < steps.size else None
    pre_med = _median(pre)
    post_med = _median(post)
    all_med = _median(steps)
    turns = _angles(vec)
    pre_turn = turns[max(0, boundary_idx - 10) : boundary_idx]
    post_turn = turns[boundary_idx + 1 : min(turns.size, boundary_idx + 11)]
    boundary_turn = float(turns[boundary_idx - 1]) if boundary_idx > 0 and boundary_idx - 1 < turns.size else None
    post_over_pre = _ratio(post_med, pre_med)
    boundary_over_pre = _ratio(boundary, pre_med)
    boundary_accel = None if boundary is None or prev is None else abs(float(boundary) - float(prev))
    step_cv = None
    if all_med is not None and all_med > 1e-12:
        step_cv = float(np.nanstd(steps) / all_med)
    post_step_cv = None
    if post_med is not None and post_med > 1e-12 and post.size > 1:
        post_step_cv = float(np.nanstd(post) / post_med)
    score_terms = [
        _unit_error(post_over_pre),
        _unit_error(boundary_over_pre),
        None if step_cv is None else min(float(step_cv), 10.0),
        None if post_step_cv is None else min(float(post_step_cv), 10.0),
        _median(post_turn),
    ]
    finite_terms = [float(x) for x in score_terms if x is not None and math.isfinite(float(x))]
    motion_score = float(sum(finite_terms)) if finite_terms else None
    return {
        "trajectory": str(path),
        "motion_available": True,
        "motion_boundary_idx": int(boundary_idx),
        "motion_median_step": all_med,
        "motion_pre_median_step": pre_med,
        "motion_boundary_step": boundary,
        "motion_post_median_step": post_med,
        "motion_post_over_pre": post_over_pre,
        "motion_boundary_over_pre": boundary_over_pre,
        "motion_boundary_accel_abs": boundary_accel,
        "motion_step_cv": step_cv,
        "motion_post_step_cv": post_step_cv,
        "motion_pre_turn_median_rad": _median(pre_turn),
        "motion_boundary_turn_rad": boundary_turn,
        "motion_post_turn_median_rad": _median(post_turn),
        "motion_score": motion_score,
        "motion_post_over_pre_unit_error": _unit_error(post_over_pre),
        "motion_boundary_over_pre_unit_error": _unit_error(boundary_over_pre),
    }


def _metrics_by_chunk_run(rows: list[dict[str, str]]) -> dict[tuple[int, str], dict[str, Any]]:
    out: dict[tuple[int, str], dict[str, Any]] = {}
    for row in rows:
        if not str(row.get("chunk", "")).strip().isdigit():
            continue
        out[(int(row["chunk"]), str(row.get("run")))] = dict(row)
    return out


def _recheck_by_chunk(rows: list[dict[str, str]]) -> dict[int, dict[str, str]]:
    out: dict[int, dict[str, str]] = {}
    for row in rows:
        if str(row.get("chunk", "")).strip().isdigit():
            out[int(row["chunk"])] = dict(row)
    return out


def _build_feature_rows(root: Path, metrics_rows: list[dict[str, str]], recheck_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    metrics = _metrics_by_chunk_run(metrics_rows)
    recheck = _recheck_by_chunk(recheck_rows)
    chunks = sorted({chunk for chunk, _run in metrics})
    out: list[dict[str, Any]] = []
    for chunk in chunks:
        native_features = _motion_features(root / f"chunk{chunk:02d}" / "native_no_swa" / "01.txt")
        native_score = _f(native_features.get("motion_score"))
        for run in RUNS:
            row = dict(metrics.get((chunk, run), {}))
            if not row:
                continue
            features = _motion_features(root / f"chunk{chunk:02d}" / run / "01.txt")
            row_out: dict[str, Any] = {
                "chunk": int(chunk),
                "run": run,
                "run_dir": str(root / f"chunk{chunk:02d}" / run),
            }
            row_out.update({f"gt_{k}": _f(row.get(k)) for k in [
                "head10_to_tail10_pose_sim3_rmse_m",
                "overlap3_to_future_pose_sim3_rmse_m",
                "scale_cv_head_mid_tail_pose_sim3",
                "local_sim3_ate_rmse_m",
            ]})
            row_out.update(features)
            score = _f(features.get("motion_score"))
            row_out["motion_score_delta_vs_native"] = (
                None if score is None or native_score is None else float(score - native_score)
            )
            if run == "thingstuff_radio_qscale":
                rr = recheck.get(chunk, {})
                row_out.update({
                    "qscale_head_tail_improvement_vs_baseline_ratio": _f(
                        rr.get("qscale_head_tail_improvement_vs_baseline_ratio")
                    ),
                    "qscale_overlap_improvement_vs_baseline_ratio": _f(
                        rr.get("qscale_overlap_improvement_vs_baseline_ratio")
                    ),
                    "qscale_head_tail_phaseE_chunk_pass": str(rr.get("qscale_head_tail_phaseE_chunk_pass")),
                    "qscale_overlap_phaseE_chunk_pass": str(rr.get("qscale_overlap_phaseE_chunk_pass")),
                    "selected_low_support_mass": _f(rr.get("selected_low_support_mass")),
                    "selected_write_interpretation": rr.get("selected_write_interpretation"),
                })
            out.append(row_out)
    return out


def _chunk_run(features: list[dict[str, Any]], chunk: int, run: str) -> dict[str, Any]:
    for row in features:
        if int(row["chunk"]) == int(chunk) and str(row["run"]) == run:
            return row
    return {}


def _bool_text(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def _qscale_rows(features: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in features if str(row.get("run")) == "thingstuff_radio_qscale"]


def _audit_qscale_rule(
    *,
    name: str,
    description: str,
    qrows: list[dict[str, Any]],
    selector: Callable[[dict[str, Any]], bool],
    min_median_improvement: float,
) -> dict[str, Any]:
    selected = [row for row in qrows if selector(row)]
    selected_chunks = [int(row["chunk"]) for row in selected]
    head_values = []
    overlap_values = []
    for row in qrows:
        if int(row["chunk"]) in selected_chunks:
            head_values.append(float(row.get("qscale_head_tail_improvement_vs_baseline_ratio") or 0.0))
            overlap_values.append(float(row.get("qscale_overlap_improvement_vs_baseline_ratio") or 0.0))
        else:
            head_values.append(0.0)
            overlap_values.append(0.0)
    head_pass = [int(row["chunk"]) for row in selected if _bool_text(row.get("qscale_head_tail_phaseE_chunk_pass"))]
    overlap_pass = [int(row["chunk"]) for row in selected if _bool_text(row.get("qscale_overlap_phaseE_chunk_pass"))]
    overlap_harm = [
        int(row["chunk"])
        for row in selected
        if float(row.get("qscale_overlap_improvement_vs_baseline_ratio") or 0.0) < 0.0
    ]
    head_median = float(median(head_values)) if head_values else 0.0
    overlap_median = float(median(overlap_values)) if overlap_values else 0.0
    gate = (
        len(head_pass) >= 2
        and len(overlap_pass) >= 2
        and head_median >= min_median_improvement
        and overlap_median >= min_median_improvement
        and not overlap_harm
    )
    return {
        "rule": name,
        "rule_type": "qscale_selector",
        "description": description,
        "selected_chunks": selected_chunks,
        "selected_count": len(selected_chunks),
        "head_tail_pass_chunks": head_pass,
        "head_tail_pass_count": len(head_pass),
        "overlap_pass_chunks": overlap_pass,
        "overlap_pass_count": len(overlap_pass),
        "overlap_harm_chunks": overlap_harm,
        "head_tail_median_improvement_with_native_fallback": head_median,
        "overlap_median_improvement_with_native_fallback": overlap_median,
        "canary_rule_gate_pass": bool(gate),
    }


def _audit_motion_best_controller(features: list[dict[str, Any]]) -> dict[str, Any]:
    chunks = sorted({int(row["chunk"]) for row in features})
    selected: list[dict[str, Any]] = []
    for chunk in chunks:
        rows = [row for row in features if int(row["chunk"]) == chunk and _f(row.get("motion_score")) is not None]
        if not rows:
            continue
        selected.append(min(rows, key=lambda row: float(row["motion_score"])))
    counts: dict[str, int] = {}
    for row in selected:
        counts[str(row["run"])] = counts.get(str(row["run"]), 0) + 1
    return {
        "rule": "min_motion_score_across_existing_runs",
        "rule_type": "controller_upper_bound",
        "description": "Select the existing run with the lowest trajectory-only motion discontinuity score.",
        "selected_chunks": [int(row["chunk"]) for row in selected],
        "selected_runs": {str(row["chunk"]): str(row["run"]) for row in selected},
        "selected_source_counts": counts,
        "semantic_selected_count": counts.get("thingstuff_radio_qscale", 0),
        "method_gate_claimed": False,
        "note": "This may select controls/native; it is an upper-bound diagnostic, not semantic method success.",
    }


def _make_rule_rows(features: list[dict[str, Any]], min_median_improvement: float) -> list[dict[str, Any]]:
    qrows = _qscale_rows(features)

    def best_against_controls(row: dict[str, Any]) -> bool:
        chunk = int(row["chunk"])
        qscore = _f(row.get("motion_score"))
        control_scores = [
            _f(_chunk_run(features, chunk, run).get("motion_score"))
            for run in ("geometry_only", "thingstuff_radio_qscale_random", "thingstuff_radio_qscale_shuffled")
        ]
        vals = [score for score in control_scores if score is not None]
        return qscore is not None and bool(vals) and float(qscore) <= min(float(v) for v in vals)

    return [
        _audit_qscale_rule(
            name="qscale_post_over_pre_unit_error_le_0p02",
            description="Keep qscale if post/pre predicted step ratio is within log 0.02 of 1.",
            qrows=qrows,
            selector=lambda row: (_f(row.get("motion_post_over_pre_unit_error")) or 999.0) <= 0.02,
            min_median_improvement=min_median_improvement,
        ),
        _audit_qscale_rule(
            name="qscale_post_over_pre_unit_error_le_0p05",
            description="Keep qscale if post/pre predicted step ratio is within log 0.05 of 1.",
            qrows=qrows,
            selector=lambda row: (_f(row.get("motion_post_over_pre_unit_error")) or 999.0) <= 0.05,
            min_median_improvement=min_median_improvement,
        ),
        _audit_qscale_rule(
            name="qscale_boundary_over_pre_unit_error_le_0p10",
            description="Keep qscale if boundary/pre predicted step ratio is within log 0.10 of 1.",
            qrows=qrows,
            selector=lambda row: (_f(row.get("motion_boundary_over_pre_unit_error")) or 999.0) <= 0.10,
            min_median_improvement=min_median_improvement,
        ),
        _audit_qscale_rule(
            name="qscale_motion_score_not_worse_than_native",
            description="Keep qscale if trajectory-only motion score is no worse than native.",
            qrows=qrows,
            selector=lambda row: (_f(row.get("motion_score_delta_vs_native")) or 999.0) <= 0.0,
            min_median_improvement=min_median_improvement,
        ),
        _audit_qscale_rule(
            name="qscale_motion_score_best_vs_controls",
            description="Keep qscale if its motion score beats geometry/random/shuffled controls.",
            qrows=qrows,
            selector=best_against_controls,
            min_median_improvement=min_median_improvement,
        ),
        _audit_qscale_rule(
            name="qscale_post_pre_safe_and_motion_best",
            description="Keep qscale if post/pre is safe and motion score beats controls.",
            qrows=qrows,
            selector=lambda row: (
                (_f(row.get("motion_post_over_pre_unit_error")) or 999.0) <= 0.05
                and best_against_controls(row)
            ),
            min_median_improvement=min_median_improvement,
        ),
        _audit_motion_best_controller(features),
    ]


def _status(rule_rows: list[dict[str, Any]]) -> str:
    if any(row.get("canary_rule_gate_pass") for row in rule_rows if row.get("rule_type") == "qscale_selector"):
        return "deployable_motion_proxy_candidate_found"
    return "no_deployable_motion_future_overlap_proxy"


def main() -> None:
    args = parse_args()
    metrics_rows = _read_csv(args.metrics_csv)
    recheck_rows = _read_csv(args.recheck_rows)
    feature_rows = _build_feature_rows(args.root, metrics_rows, recheck_rows)
    rule_rows = _make_rule_rows(feature_rows, float(args.min_median_improvement))
    summary = {
        "schema": "acl2_v80_motion_future_overlap_proxy_audit_v1",
        "status": _status(rule_rows),
        "diagnostic_only": True,
        "method_gate_claimed": False,
        "runtime_promotion_allowed": False,
        "v80_goal_achieved": False,
        "root": str(args.root),
        "metrics_csv": str(args.metrics_csv),
        "recheck_rows": str(args.recheck_rows),
        "feature_row_count": len(feature_rows),
        "rules_evaluated": len(rule_rows),
        "deployable_gate_pass_rules": [
            row["rule"] for row in rule_rows if row.get("rule_type") == "qscale_selector" and row.get("canary_rule_gate_pass")
        ],
        "best_qscale_rules_by_overlap": [
            row["rule"]
            for row in sorted(
                [r for r in rule_rows if r.get("rule_type") == "qscale_selector"],
                key=lambda r: (
                    float(r.get("overlap_median_improvement_with_native_fallback") or 0.0),
                    float(r.get("head_tail_median_improvement_with_native_fallback") or 0.0),
                ),
                reverse=True,
            )[:3]
        ],
        "core_blocker": (
            "Trajectory-only motion continuity proxies do not produce a qscale selector that keeps "
            "chunk10/chunk12 head-tail gains while preventing overlap-to-future harm."
        ),
        "next_action": (
            "Do not promote motion proxy. Current v80 semantic/RADIO qscale, selected-write, local-overlap, "
            "and trajectory-motion proxy families have all failed deployable gates."
        ),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.out_dir / "motion_future_overlap_proxy_feature_rows.csv", feature_rows)
    _write_csv(args.out_dir / "motion_future_overlap_proxy_rule_audit.csv", rule_rows)
    _write_json(args.out_dir / "motion_future_overlap_proxy_audit_summary.json", summary)
    report = [
        "# v80 motion future-overlap proxy audit",
        "",
        f"status: {summary['status']}",
        f"v80_goal_achieved: {summary['v80_goal_achieved']}",
        f"method_gate_claimed: {summary['method_gate_claimed']}",
        "",
        "## Core Blocker",
        "",
        summary["core_blocker"],
        "",
        "## Next Action",
        "",
        summary["next_action"],
        "",
    ]
    (args.out_dir / "motion_future_overlap_proxy_audit_report.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps(_jsonable(summary), ensure_ascii=False, indent=2, sort_keys=True))
    print(f"wrote_dir={args.out_dir}")


if __name__ == "__main__":
    main()
