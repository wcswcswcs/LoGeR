#!/usr/bin/env python3
"""Evaluate v102 state-machine trace/action runs with trajectory metrics.

The evaluator reuses the repository's TUM/KITTI Sim(3) trajectory diagnostics
and adds v102 case metadata.  It does not promote trace-only runs to Stage4
success; it only materializes L1/L2/L3 evidence for later action-surface audits.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.evaluate_v68_phaseD_read_smoke import _eval_run, _load_kitti_gt  # noqa: E402


RESULT_ROOT = Path("results/acl2_v102tf_drift_source_autopsy_semantic_oracle_memory_control")
DEFAULT_TARGET_CSV = RESULT_ROOT / "stage4_memory_action_surface_oracle/v102_state_machine_scaffold_trace_targets.csv"
DEFAULT_RUN_ROOT = RESULT_ROOT / "stage4_memory_action_surface_oracle/v102_state_machine_scaffold_trace_delay_update_v2"
DEFAULT_OUT = RESULT_ROOT / "stage4_memory_action_surface_oracle/state_machine_trace_run_metrics"
DEFAULT_KITTI_GT_ROOT = Path("/mnt/data/users/chengshun.wang/data/kitti_odometry/dataset/poses")

LOWER_IS_BETTER_KEYS = [
    "local_sim3_ate_rmse_m",
    "head10_to_tail10_pose_sim3_rmse_m",
    "overlap3_to_future_pose_sim3_rmse_m",
    "scale_cv_head_mid_tail_pose_sim3",
]


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    if not keys:
        keys = ["empty"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            clean: dict[str, Any] = {}
            for key in keys:
                value = row.get(key, "")
                if isinstance(value, (list, tuple, dict)):
                    clean[key] = json.dumps(jsonable(value), ensure_ascii=False, sort_keys=True)
                else:
                    clean[key] = value
            writer.writerow(clean)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def fnum(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return math.nan
    return out if math.isfinite(out) else math.nan


def bval(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def rel_improvement(base: Any, candidate: Any) -> float:
    base_f = fnum(base)
    cand_f = fnum(candidate)
    if not math.isfinite(base_f) or not math.isfinite(cand_f) or abs(base_f) < 1e-12:
        return math.nan
    return (base_f - cand_f) / (abs(base_f) + 1e-12)


def role(row: dict[str, Any]) -> str:
    text = str(row.get("ambiguous_or_control_role") or row.get("case_label") or "")
    if text:
        return text
    if bval(row.get("strict_clean_handoff_positive")):
        return "strict_clean_handoff_positive"
    return "unknown"


def gt_for_seq(gt_root: Path, seq: str) -> Path:
    path = gt_root / f"{int(seq):02d}.txt"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def evaluate_case(row: dict[str, str], *, run_root: Path, gt_root: Path, variant: str) -> dict[str, Any]:
    case_id = str(row.get("case_id", "")).strip()
    seq = f"{int(row.get('seq', '0')):02d}"
    run_dir = run_root / case_id / variant
    gt_path = gt_for_seq(gt_root, seq)
    out: dict[str, Any] = dict(row)
    out.update({
        "variant": variant,
        "run_dir": run_dir.as_posix(),
        "trajectory_name": f"{seq}.txt",
        "trajectory_path": (run_dir / f"{seq}.txt").as_posix(),
        "gt_path": gt_path.as_posix(),
        "eval_status": "not_run",
        "eval_error": "",
    })
    try:
        _, gt_poses_all, gt_pos_all = _load_kitti_gt(gt_path)
        metrics = _eval_run(f"{case_id}_{variant}", run_dir, gt_poses_all, gt_pos_all, trajectory_name=f"{seq}.txt")
    except Exception as exc:  # noqa: BLE001
        out["eval_status"] = "failed"
        out["eval_error"] = f"{type(exc).__name__}: {exc}"
        return out
    out.update(metrics)
    out["eval_status"] = "ok"
    out["role"] = role(out)
    for key in LOWER_IS_BETTER_KEYS:
        atlas_key = {
            "local_sim3_ate_rmse_m": "L1_local_sim3_ate",
            "head10_to_tail10_pose_sim3_rmse_m": "L2_head_tail_proxy_error",
            "overlap3_to_future_pose_sim3_rmse_m": "",
            "scale_cv_head_mid_tail_pose_sim3": "",
        }[key]
        if atlas_key:
            out[f"delta_vs_atlas_{atlas_key}"] = fnum(out.get(key)) - fnum(out.get(atlas_key))
    return out


def attach_baseline_comparison(rows: list[dict[str, Any]], baseline_rows: list[dict[str, Any]]) -> None:
    baseline_by_case = {str(row.get("case_id", "")): row for row in baseline_rows if row.get("eval_status") == "ok"}
    for row in rows:
        base = baseline_by_case.get(str(row.get("case_id", "")))
        row["baseline_eval_status"] = base.get("eval_status", "") if base else "missing"
        if base is None:
            continue
        row["baseline_run_dir"] = base.get("run_dir", "")
        for key in LOWER_IS_BETTER_KEYS:
            row[f"baseline_{key}"] = base.get(key)
            row[f"delta_vs_baseline_{key}"] = fnum(row.get(key)) - fnum(base.get(key))
            row[f"relative_improvement_vs_baseline_{key}"] = rel_improvement(base.get(key), row.get(key))


def summarize(
    rows: list[dict[str, Any]],
    *,
    target_csv: Path,
    run_root: Path,
    variant: str,
    baseline_run_root: Path | None = None,
) -> dict[str, Any]:
    ok = [row for row in rows if row.get("eval_status") == "ok"]
    roles = sorted({str(row.get("role", "")) for row in ok if row.get("role")})
    summary: dict[str, Any] = {
        "schema": "acl2_v102_state_machine_trace_run_metrics_v1",
        "target_csv": target_csv.as_posix(),
        "run_root": run_root.as_posix(),
        "baseline_run_root": baseline_run_root.as_posix() if baseline_run_root else "",
        "variant": variant,
        "case_count": len(rows),
        "ok_case_count": len(ok),
        "failed_case_count": len(rows) - len(ok),
        "roles_seen": roles,
        "diagnostic_only": True,
        "runtime_action_allowed": False,
        "stage4_strict_memory_action_surface_pass": False,
        "metric_keys": LOWER_IS_BETTER_KEYS,
    }
    for key in LOWER_IS_BETTER_KEYS:
        vals = [fnum(row.get(key)) for row in ok]
        vals = [v for v in vals if math.isfinite(v)]
        summary[f"{key}_mean"] = float(np.mean(vals)) if vals else None
        summary[f"{key}_median"] = float(np.median(vals)) if vals else None
    for group in roles:
        group_rows = [row for row in ok if row.get("role") == group]
        summary[f"{group}_case_count"] = len(group_rows)
        for key in LOWER_IS_BETTER_KEYS:
            vals = [fnum(row.get(key)) for row in group_rows]
            vals = [v for v in vals if math.isfinite(v)]
            summary[f"{group}_{key}_median"] = float(np.median(vals)) if vals else None
    if baseline_run_root is not None:
        paired = [
            row for row in ok
            if str(row.get("baseline_eval_status", "")) == "ok"
        ]
        summary["paired_baseline_case_count"] = len(paired)
        for key in LOWER_IS_BETTER_KEYS:
            vals = [fnum(row.get(f"relative_improvement_vs_baseline_{key}")) for row in paired]
            vals = [v for v in vals if math.isfinite(v)]
            summary[f"relative_improvement_vs_baseline_{key}_mean"] = float(np.mean(vals)) if vals else None
            summary[f"relative_improvement_vs_baseline_{key}_median"] = float(np.median(vals)) if vals else None
            deltas = [fnum(row.get(f"delta_vs_baseline_{key}")) for row in paired]
            deltas = [v for v in deltas if math.isfinite(v)]
            summary[f"delta_vs_baseline_{key}_median"] = float(np.median(deltas)) if deltas else None
        for group in roles:
            group_rows = [row for row in paired if row.get("role") == group]
            summary[f"{group}_paired_baseline_case_count"] = len(group_rows)
            for key in LOWER_IS_BETTER_KEYS:
                vals = [fnum(row.get(f"relative_improvement_vs_baseline_{key}")) for row in group_rows]
                vals = [v for v in vals if math.isfinite(v)]
                summary[f"{group}_relative_improvement_vs_baseline_{key}_median"] = (
                    float(np.median(vals)) if vals else None
                )
    summary["conclusion"] = (
        "Trajectory metrics were materialized for the supplied v102 state-machine trace runs. "
        "This is a baseline/evaluator artifact only unless paired with a non-baseline action run "
        "and strict Stage3/Stage4 gates."
    )
    return summary


def build_report(summary: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    lines = [
        "# v102 State-Machine Trace Run Metrics",
        "",
        f"- run_root: `{summary.get('run_root')}`",
        f"- baseline_run_root: `{summary.get('baseline_run_root')}`",
        f"- variant: `{summary.get('variant')}`",
        f"- ok_case_count: {summary.get('ok_case_count')}/{summary.get('case_count')}",
        f"- runtime_action_allowed: {summary.get('runtime_action_allowed')}",
        f"- stage4_strict_memory_action_surface_pass: {summary.get('stage4_strict_memory_action_surface_pass')}",
        "",
        "| case_id | role | local_sim3 | head_tail | overlap_future | scale_cv | rel_impr_local | rel_impr_head_tail | eval_status |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| {case_id} | {role} | {local} | {head_tail} | {overlap} | {scale_cv} | {rel_local} | {rel_head} | {status} |".format(
                case_id=row.get("case_id", ""),
                role=row.get("role", row.get("ambiguous_or_control_role", "")),
                local=row.get("local_sim3_ate_rmse_m", ""),
                head_tail=row.get("head10_to_tail10_pose_sim3_rmse_m", ""),
                overlap=row.get("overlap3_to_future_pose_sim3_rmse_m", ""),
                scale_cv=row.get("scale_cv_head_mid_tail_pose_sim3", ""),
                rel_local=row.get("relative_improvement_vs_baseline_local_sim3_ate_rmse_m", ""),
                rel_head=row.get("relative_improvement_vs_baseline_head10_to_tail10_pose_sim3_rmse_m", ""),
                status=row.get("eval_status", ""),
            )
        )
    lines.extend(["", "Conclusion:", "", str(summary.get("conclusion", ""))])
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-csv", type=Path, default=DEFAULT_TARGET_CSV)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--baseline-run-root", type=Path, default=None)
    parser.add_argument("--variant", default="READ_NO_ACTION")
    parser.add_argument("--gt-root", type=Path, default=DEFAULT_KITTI_GT_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    targets = read_csv_rows(args.target_csv)
    baseline_rows: list[dict[str, Any]] = []
    if args.baseline_run_root is not None:
        baseline_rows = [
            evaluate_case(row, run_root=args.baseline_run_root, gt_root=args.gt_root, variant=args.variant)
            for row in targets
        ]
    rows = [evaluate_case(row, run_root=args.run_root, gt_root=args.gt_root, variant=args.variant) for row in targets]
    if baseline_rows:
        attach_baseline_comparison(rows, baseline_rows)
    summary = summarize(
        rows,
        target_csv=args.target_csv,
        run_root=args.run_root,
        variant=args.variant,
        baseline_run_root=args.baseline_run_root,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "state_machine_trace_run_metrics_rows.csv", rows)
    write_json(args.out_dir / "state_machine_trace_run_metrics_summary.json", summary)
    write_text(args.out_dir / "state_machine_trace_run_metrics_report.md", build_report(summary, rows))
    print(json.dumps(jsonable(summary), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
