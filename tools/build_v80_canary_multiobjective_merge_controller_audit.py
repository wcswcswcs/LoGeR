#!/usr/bin/env python3
"""Build a diagnostic multi-chunk v80 merge/gauge controller audit.

This reuses existing canary artifacts and materializes a virtual controller
root with symlinks. Decisions use non-GT proxy features only. The output is
diagnostic-only; selecting geometry_only or native is not semantic success.
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
DEFAULT_SOURCE_ROOT = REPORT_ROOT / "phase9_seq01_thingstuff_radio_qscale_canary5_ref055_chunks006_008_010_012"
DEFAULT_METRICS = DEFAULT_SOURCE_ROOT / "thingstuff_radio_qscale_ref055_canary5_run_metrics.csv"
DEFAULT_OUT_ROOT = REPORT_ROOT / "phase9_seq01_canary5_multiobjective_merge_controller_v1"
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
    "semantic_merge_overlap_residual",
    "semantic_merge_condition_score",
    "semantic_merge_native_overlap_residual",
    "semantic_merge_final_overlap_residual",
    "semantic_merge_native_overlap_guard_rejected",
    "semantic_merge_qscale_observability",
    "semantic_merge_radio_handoff_weighted_qscale_observability",
    "online_scale_state_active",
    "online_scale_state_gate_pass",
    "online_scale_state_gate_reason",
    "online_scale_state_gate_qscale",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--metrics-csv", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--chunks", default="6,7,8,10,12")
    parser.add_argument("--controller-run", default="multiobjective_controller_v1")
    parser.add_argument("--candidate-run", default="thingstuff_radio_qscale")
    parser.add_argument("--baseline-run", default="native_no_swa")
    parser.add_argument("--geometry-run", default="geometry_only")
    parser.add_argument("--random-run", default="thingstuff_radio_qscale_random")
    parser.add_argument("--shuffled-run", default="thingstuff_radio_qscale_shuffled")
    parser.add_argument("--qscale-random-gap-min", type=float, default=0.02)
    parser.add_argument(
        "--selection-policy",
        default="v1",
        choices=("v1", "residual_controls_post_unit"),
        help=(
            "v1 keeps the original conservative qscale/geometry proxy. "
            "residual_controls_post_unit selects a semantic candidate only when its sampled "
            "overlap residual beats random/shuffled/geometry controls and its output trajectory "
            "post/pre step ratio stays close to 1.0."
        ),
    )
    parser.add_argument("--post-over-pre-target", type=float, default=1.0)
    parser.add_argument("--post-over-pre-tolerance", type=float, default=0.02)
    parser.add_argument("--runs", default=",".join(DEFAULT_RUNS))
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _parse_chunks(text: str) -> list[int]:
    return [int(part.strip()) for part in text.split(",") if part.strip()]


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


def _selected_trace(source_root: Path, chunk: int, run: str) -> dict[str, Any]:
    rows = _read_jsonl(source_root / f"chunk{chunk:02d}" / run / "merge_state_trace.jsonl")
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


def _proxy_qscale(row: dict[str, Any]) -> float | None:
    return _safe_float(
        row.get("semantic_merge_radio_handoff_weighted_qscale_observability")
        if row.get("semantic_merge_radio_handoff_weighted_qscale_observability") is not None
        else row.get("semantic_merge_qscale_observability")
    )


def _proxy_residual(row: dict[str, Any]) -> tuple[float | None, str]:
    final_residual = _safe_float(row.get("semantic_merge_final_overlap_residual"))
    if final_residual is not None:
        return final_residual, "final_overlap_residual"
    legacy_residual = _safe_float(row.get("semantic_merge_overlap_residual"))
    if legacy_residual is not None:
        return legacy_residual, "legacy_overlap_residual"
    return None, "missing"


def _row_map(args: argparse.Namespace, chunks: list[int], runs: list[str]) -> dict[tuple[int, str], dict[str, Any]]:
    metric_rows = {}
    for row in _read_csv(args.metrics_csv):
        try:
            key = (int(row.get("chunk", "")), str(row.get("run")))
        except ValueError:
            continue
        metric_rows[key] = row
    out: dict[tuple[int, str], dict[str, Any]] = {}
    for chunk in chunks:
        for run in runs:
            row: dict[str, Any] = {
                "chunk": int(chunk),
                "run": run,
                "run_dir": str(args.source_root / f"chunk{chunk:02d}" / run),
            }
            row.update(_trajectory_features(args.source_root / f"chunk{chunk:02d}" / run))
            metrics = metric_rows.get((chunk, run), {})
            for key in METRIC_KEYS:
                row[key] = _safe_float(metrics.get(key))
            trace = _selected_trace(args.source_root, chunk, run)
            for key in TRACE_KEYS:
                row[key] = trace.get(key)
            row["proxy_qscale"] = _proxy_qscale(row)
            out[(chunk, run)] = row
    return out


def _choose_chunk(args: argparse.Namespace, rows: dict[tuple[int, str], dict[str, Any]], chunk: int) -> dict[str, Any]:
    cand = rows.get((chunk, args.candidate_run), {})
    geom = rows.get((chunk, args.geometry_run), {})
    native = rows.get((chunk, args.baseline_run), {})
    random = rows.get((chunk, args.random_run), {})
    shuffled = rows.get((chunk, args.shuffled_run), {})
    cand_q = _proxy_qscale(cand)
    random_q = _proxy_qscale(random)
    q_gap = abs(cand_q - random_q) if cand_q is not None and random_q is not None else None
    semantic_not_selective = bool(q_gap is not None and q_gap < float(args.qscale_random_gap_min))

    cand_res, cand_res_source = _proxy_residual(cand)
    geom_res, geom_res_source = _proxy_residual(geom)
    cand_post = _safe_float(cand.get("traj_post_over_pre"))
    geom_post = _safe_float(geom.get("traj_post_over_pre"))
    geom_proxy_better = bool(
        cand_res is not None
        and geom_res is not None
        and geom_res <= cand_res
        and cand_post is not None
        and geom_post is not None
        and geom_post >= cand_post
    )
    cand_proxy_safe_v1 = bool(
        not semantic_not_selective
        and cand_res is not None
        and geom_res is not None
        and cand_res <= geom_res
        and cand_post is not None
        and geom_post is not None
        and cand_post >= geom_post
    )
    control_residuals = [
        value
        for value in (
            _proxy_residual(random)[0],
            _proxy_residual(shuffled)[0],
            geom_res,
        )
        if value is not None
    ]
    candidate_beats_control_residuals = bool(
        cand_res is not None and control_residuals and cand_res < min(control_residuals)
    )
    post_unit_error = (
        abs(float(cand_post) - float(args.post_over_pre_target)) if cand_post is not None else None
    )
    post_unit_safe = bool(
        post_unit_error is not None and post_unit_error <= float(args.post_over_pre_tolerance)
    )
    cand_proxy_safe_v2 = bool(candidate_beats_control_residuals and post_unit_safe)
    cand_proxy_safe = cand_proxy_safe_v2 if args.selection_policy == "residual_controls_post_unit" else cand_proxy_safe_v1
    if cand_proxy_safe:
        selected = args.candidate_run
        reason = f"candidate_proxy_safe_{args.selection_policy}"
    elif geom_proxy_better:
        selected = args.geometry_run
        reason = "geometry_residual_and_post_proxy_better"
    else:
        selected = args.baseline_run
        reason = "native_fallback_no_deployable_proxy_win"
    return {
        "chunk": int(chunk),
        "selected_source_run": selected,
        "selection_reason": reason,
        "candidate_proxy_safe": cand_proxy_safe,
        "candidate_proxy_safe_v1": cand_proxy_safe_v1,
        "candidate_proxy_safe_v2": cand_proxy_safe_v2,
        "geometry_proxy_better": geom_proxy_better,
        "selection_policy": args.selection_policy,
        "semantic_qscale_not_selective": semantic_not_selective,
        "candidate_random_qscale_abs_gap": q_gap,
        "candidate_proxy_qscale": cand_q,
        "random_proxy_qscale": random_q,
        "candidate_final_overlap_residual": cand_res,
        "geometry_final_overlap_residual": geom_res,
        "candidate_beats_control_residuals": candidate_beats_control_residuals,
        "best_control_final_overlap_residual": min(control_residuals) if control_residuals else None,
        "candidate_residual_source": cand_res_source,
        "geometry_residual_source": geom_res_source,
        "candidate_traj_post_over_pre": cand_post,
        "geometry_traj_post_over_pre": geom_post,
        "candidate_post_over_pre_unit_error": post_unit_error,
        "candidate_post_over_pre_unit_safe": post_unit_safe,
        "post_over_pre_target": float(args.post_over_pre_target),
        "post_over_pre_tolerance": float(args.post_over_pre_tolerance),
        "selected_head_tail": rows.get((chunk, selected), {}).get("head10_to_tail10_pose_sim3_rmse_m"),
        "selected_overlap": rows.get((chunk, selected), {}).get("overlap3_to_future_pose_sim3_rmse_m"),
        "native_head_tail": native.get("head10_to_tail10_pose_sim3_rmse_m"),
        "native_overlap": native.get("overlap3_to_future_pose_sim3_rmse_m"),
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


def _materialize(args: argparse.Namespace, decisions: list[dict[str, Any]], runs: list[str]) -> None:
    for decision in decisions:
        chunk = int(decision["chunk"])
        dst_chunk = args.out_root / f"chunk{chunk:02d}"
        for run in runs:
            _replace_symlink(dst_chunk / run, args.source_root / f"chunk{chunk:02d}" / run, dry_run=args.dry_run)
        _replace_symlink(
            dst_chunk / args.controller_run,
            args.source_root / f"chunk{chunk:02d}" / str(decision["selected_source_run"]),
            dry_run=args.dry_run,
        )


def _counts(decisions: list[dict[str, Any]]) -> dict[str, int]:
    out: dict[str, int] = {}
    for decision in decisions:
        key = str(decision["selected_source_run"])
        out[key] = out.get(key, 0) + 1
    return out


def main() -> None:
    args = _parse_args()
    chunks = _parse_chunks(args.chunks)
    runs = [part.strip() for part in args.runs.split(",") if part.strip()]
    rows_by_key = _row_map(args, chunks, runs)
    rows = list(rows_by_key.values())
    decisions = [_choose_chunk(args, rows_by_key, chunk) for chunk in chunks]
    _materialize(args, decisions, runs)
    selected_counts = _counts(decisions)
    semantic_selected_count = int(selected_counts.get(args.candidate_run, 0))
    summary = {
        "schema": "acl2_v80_canary_multiobjective_merge_controller_audit_v1",
        "status": "diagnostic_virtual_controller_materialized" if not args.dry_run else "diagnostic_dry_run",
        "source_root": str(args.source_root),
        "out_root": str(args.out_root),
        "metrics_csv": str(args.metrics_csv),
        "chunks": chunks,
        "controller_run": args.controller_run,
        "selected_source_counts": selected_counts,
        "semantic_selected_count": semantic_selected_count,
        "diagnostic_only": True,
        "method_gate_claimed": False,
        "runtime_promotion_allowed": False,
        "v80_goal_achieved": False,
        "decisions": decisions,
        "rows": rows,
    }
    args.out_root.mkdir(parents=True, exist_ok=True)
    _write_json(args.out_root / "canary_multiobjective_controller_decision.json", summary)
    _write_csv(args.out_root / "canary_multiobjective_controller_decisions.csv", decisions)
    _write_csv(args.out_root / "canary_multiobjective_controller_proxy_rows.csv", rows)
    report = [
        "# ACL2 v80 canary multi-objective merge/gauge controller audit",
        "",
        "## Decision Summary",
        "",
        f"- selected_source_counts: `{selected_counts}`",
        f"- semantic_selected_count: `{semantic_selected_count}`",
        f"- diagnostic_only: `{summary['diagnostic_only']}`",
        f"- method_gate_claimed: `{summary['method_gate_claimed']}`",
        "",
        "## Per-Chunk Decisions",
        "",
    ]
    for decision in decisions:
        report.append(
            f"- chunk{int(decision['chunk']):02d}: `{decision['selected_source_run']}` "
            f"({decision['selection_reason']})"
        )
    report.extend(
        [
            "",
            "Selecting geometry_only or native is an explicit safety fallback and not semantic-method success.",
            "",
        ]
    )
    (args.out_root / "canary_multiobjective_controller_report.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps(_jsonable({k: v for k, v in summary.items() if k != "rows"}), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
