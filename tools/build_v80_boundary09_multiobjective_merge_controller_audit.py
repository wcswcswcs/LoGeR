#!/usr/bin/env python3
"""Build a diagnostic multi-objective merge/gauge controller audit for v80.

The controller is virtual and read-only with respect to model outputs: it
chooses among already-materialized run directories using non-GT proxy features
from merge_state_trace plus trajectory shape diagnostics. It then materializes a
symlink root so the standard PhaseE evaluator can score the virtual controller.

This script does not claim semantic success. Selecting geometry_only is an
explicit fallback diagnosis, not a semantic-method pass.
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
DEFAULT_SOURCE_ROOT = REPORT_ROOT / "phase9_seq01_thingstuff_radio_qscale_boundary09_alpha160_qscale_weighted_gate_chunks009"
DEFAULT_METRICS = DEFAULT_SOURCE_ROOT / "thingstuff_radio_qscale_boundary09_alpha160_qscale_weighted_gate_run_metrics.csv"
DEFAULT_OUT_ROOT = REPORT_ROOT / "phase9_seq01_boundary09_multiobjective_merge_controller_v1"
DEFAULT_RUNS = (
    "native_no_swa",
    "thingstuff_radio_qscale",
    "thingstuff_radio_qscale_random",
    "thingstuff_radio_qscale_shuffled",
    "geometry_only",
)
PROXY_KEYS = (
    "semantic_merge_strategy",
    "semantic_merge_control_type",
    "semantic_merge_scale",
    "semantic_merge_candidate_scale",
    "semantic_merge_blend_scale",
    "semantic_merge_overlap_residual",
    "semantic_merge_condition_score",
    "semantic_merge_native_overlap_residual",
    "semantic_merge_final_overlap_residual",
    "semantic_merge_native_overlap_guard_rejected",
    "semantic_merge_radio_handoff_weighted_qscale_observability",
    "online_scale_state_active",
    "online_scale_state_gate_pass",
    "online_scale_state_gate_reason",
    "online_scale_state_gate_qscale",
)
METRIC_KEYS = (
    "head10_to_tail10_pose_sim3_rmse_m",
    "overlap3_to_future_pose_sim3_rmse_m",
    "local_sim3_ate_rmse_m",
    "scale_cv_head_mid_tail_pose_sim3",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--metrics-csv", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--controller-run", default="multiobjective_controller_v1")
    parser.add_argument("--candidate-run", default="thingstuff_radio_qscale")
    parser.add_argument("--baseline-run", default="native_no_swa")
    parser.add_argument("--geometry-run", default="geometry_only")
    parser.add_argument("--random-run", default="thingstuff_radio_qscale_random")
    parser.add_argument("--shuffled-run", default="thingstuff_radio_qscale_shuffled")
    parser.add_argument("--qscale-random-gap-min", type=float, default=0.02)
    parser.add_argument("--runs", default=",".join(DEFAULT_RUNS))
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


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


def _selected_trace(source_root: Path, run: str) -> dict[str, Any]:
    rows = _read_jsonl(source_root / "chunk09" / run / "merge_state_trace.jsonl")
    selected = [row for row in rows if int(row.get("local_chunk_idx", -1)) == 1]
    return selected[-1] if selected else (rows[-1] if rows else {})


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


def _row_by_run(args: argparse.Namespace) -> dict[str, dict[str, Any]]:
    metric_rows = {str(row.get("run")): row for row in _read_csv(args.metrics_csv)}
    out: dict[str, dict[str, Any]] = {}
    for run in [part.strip() for part in args.runs.split(",") if part.strip()]:
        row: dict[str, Any] = {"run": run, "run_dir": str(args.source_root / "chunk09" / run)}
        row.update(_trajectory_features(args.source_root / "chunk09" / run))
        metrics = metric_rows.get(run, {})
        for key in METRIC_KEYS:
            row[key] = _safe_float(metrics.get(key))
        trace = _selected_trace(args.source_root, run)
        for key in PROXY_KEYS:
            row[key] = trace.get(key)
        out[run] = row
    return out


def _finite(value: Any, default: float) -> float:
    out = _safe_float(value)
    return float(out) if out is not None else float(default)


def _choose_source(args: argparse.Namespace, rows: dict[str, dict[str, Any]]) -> dict[str, Any]:
    cand = rows.get(args.candidate_run, {})
    geom = rows.get(args.geometry_run, {})
    native = rows.get(args.baseline_run, {})
    random = rows.get(args.random_run, {})

    cand_wq = _safe_float(cand.get("semantic_merge_radio_handoff_weighted_qscale_observability"))
    random_wq = _safe_float(random.get("semantic_merge_radio_handoff_weighted_qscale_observability"))
    qscale_gap = (
        abs(float(cand_wq) - float(random_wq))
        if cand_wq is not None and random_wq is not None
        else None
    )
    semantic_false_positive = bool(
        qscale_gap is not None
        and qscale_gap < float(args.qscale_random_gap_min)
        and bool(cand.get("online_scale_state_active"))
        and bool(random.get("online_scale_state_active"))
    )

    cand_res = _safe_float(cand.get("semantic_merge_final_overlap_residual"))
    geom_res = _safe_float(geom.get("semantic_merge_final_overlap_residual"))
    cand_post = _safe_float(cand.get("traj_post_over_pre"))
    geom_post = _safe_float(geom.get("traj_post_over_pre"))
    geom_proxy_better = bool(
        geom_res is not None
        and cand_res is not None
        and geom_res <= cand_res
        and geom_post is not None
        and cand_post is not None
        and geom_post >= cand_post
    )
    cand_proxy_safe = bool(
        not semantic_false_positive
        and cand_res is not None
        and geom_res is not None
        and cand_res <= geom_res
        and cand_post is not None
        and geom_post is not None
        and cand_post >= geom_post
    )

    if cand_proxy_safe:
        source = args.candidate_run
        reason = "candidate_proxy_safe"
    elif geom_proxy_better:
        source = args.geometry_run
        reason = "geometry_residual_and_post_proxy_better"
    else:
        source = args.baseline_run
        reason = "native_fallback_no_deployable_proxy_win"

    return {
        "controller_run": args.controller_run,
        "selected_source_run": source,
        "selection_reason": reason,
        "diagnostic_only": True,
        "method_gate_claimed": False,
        "runtime_promotion_allowed": False,
        "semantic_false_positive": semantic_false_positive,
        "candidate_random_weighted_qscale_abs_gap": qscale_gap,
        "qscale_random_gap_min": float(args.qscale_random_gap_min),
        "candidate_final_overlap_residual": cand_res,
        "geometry_final_overlap_residual": geom_res,
        "candidate_traj_post_over_pre": cand_post,
        "geometry_traj_post_over_pre": geom_post,
        "candidate_proxy_safe": cand_proxy_safe,
        "geometry_proxy_better": geom_proxy_better,
        "selected_source_metrics": {
            key: rows.get(source, {}).get(key)
            for key in METRIC_KEYS
        },
        "candidate_metrics": {key: cand.get(key) for key in METRIC_KEYS},
        "geometry_metrics": {key: geom.get(key) for key in METRIC_KEYS},
        "native_metrics": {key: native.get(key) for key in METRIC_KEYS},
    }


def _replace_symlink(link: Path, target: Path, *, dry_run: bool) -> None:
    if dry_run:
        return
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.is_symlink() or not link.exists():
        if link.exists() or link.is_symlink():
            link.unlink()
        link.symlink_to(target.resolve(), target_is_directory=True)
        return
    raise FileExistsError(f"Refusing to replace non-symlink path: {link}")


def _materialize(args: argparse.Namespace, selected_source: str) -> None:
    chunk_dst = args.out_root / "chunk09"
    for run in [part.strip() for part in args.runs.split(",") if part.strip()]:
        _replace_symlink(chunk_dst / run, args.source_root / "chunk09" / run, dry_run=args.dry_run)
    _replace_symlink(
        chunk_dst / args.controller_run,
        args.source_root / "chunk09" / selected_source,
        dry_run=args.dry_run,
    )


def main() -> None:
    args = _parse_args()
    rows_by_run = _row_by_run(args)
    decision = _choose_source(args, rows_by_run)
    _materialize(args, str(decision["selected_source_run"]))

    rows = list(rows_by_run.values())
    summary = {
        "schema": "acl2_v80_boundary09_multiobjective_merge_controller_audit_v1",
        "status": "diagnostic_virtual_controller_materialized" if not args.dry_run else "diagnostic_dry_run",
        "source_root": str(args.source_root),
        "out_root": str(args.out_root),
        "metrics_csv": str(args.metrics_csv),
        "decision": decision,
        "rows": rows,
        "v80_goal_achieved": False,
    }
    args.out_root.mkdir(parents=True, exist_ok=True)
    _write_json(args.out_root / "multiobjective_controller_decision.json", summary)
    _write_csv(args.out_root / "multiobjective_controller_proxy_rows.csv", rows)
    report = [
        "# ACL2 v80 boundary09 multi-objective merge/gauge controller audit",
        "",
        "## Decision",
        "",
        f"- selected_source_run: `{decision['selected_source_run']}`",
        f"- selection_reason: `{decision['selection_reason']}`",
        f"- diagnostic_only: `{decision['diagnostic_only']}`",
        f"- method_gate_claimed: `{decision['method_gate_claimed']}`",
        f"- semantic_false_positive: `{decision['semantic_false_positive']}`",
        f"- candidate_random_weighted_qscale_abs_gap: `{decision['candidate_random_weighted_qscale_abs_gap']}`",
        f"- candidate_final_overlap_residual: `{decision['candidate_final_overlap_residual']}`",
        f"- geometry_final_overlap_residual: `{decision['geometry_final_overlap_residual']}`",
        f"- candidate_traj_post_over_pre: `{decision['candidate_traj_post_over_pre']}`",
        f"- geometry_traj_post_over_pre: `{decision['geometry_traj_post_over_pre']}`",
        "",
        "## Interpretation",
        "",
        "This virtual controller selects geometry_only as a safety fallback when semantic qscale is indistinguishable from same-mass random and geometry residual/post-boundary proxies are better. This is not semantic-method success because geometry_only remains a required control.",
        "",
    ]
    (args.out_root / "multiobjective_controller_report.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps(_jsonable({k: v for k, v in summary.items() if k != "rows"}), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
