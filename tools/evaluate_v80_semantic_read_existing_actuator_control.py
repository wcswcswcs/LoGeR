#!/usr/bin/env python3
"""Evaluate ACL2 v80 Phase3 short READ existing-actuator action/control runs."""

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


DEFAULT_BASE_DIR = Path(
    "results/kitti01_hmc_v2/acl2_v80tf_multiseq_goodbad_semantic_three_memory_control/"
    "report_final/phase3_short_read_existing_actuator_control/rollouts"
)
DEFAULT_TARGET_CSV = Path(
    "results/kitti01_hmc_v2/acl2_v80tf_multiseq_goodbad_semantic_three_memory_control/"
    "report_final/phase1_three_memory_case_bank/short_single_chunk_cases.csv"
)
DEFAULT_GT_ROOT = Path("/mnt/data/users/chengshun.wang/data/kitti_odometry/dataset/poses")
DEFAULT_CANDIDATES = ["READ1_EXISTING_L07_LAYOUT_SELECT"]
DEFAULT_CONTROLS = [
    "READ7_GEOMETRY_ONLY_CONTROL",
    "READ8_LABEL_SHUFFLE",
    "READ9_CONFIDENCE_SHUFFLE",
    "READ10_SAME_READ_MASS_RANDOM",
    "READ11_GROUP_STRATIFIED_RANDOM",
]
BASELINE = "READ0_NATIVE"
DEDICATED_QK_PAIR_CASE_MARKERS = ("DEDICATED_QK_PAIR", "QK_KEYSTABLE")
LOWER_IS_BETTER_KEYS = [
    "local_sim3_ate_rmse_m",
    "head10_to_tail10_pose_sim3_rmse_m",
    "scale_cv_head_mid_tail_pose_sim3",
]
PLAN_THRESHOLD_BY_KEY = {
    "J_short_eval_proxy": 0.05,
    "local_sim3_ate_rmse_m": 0.10,
    "head10_to_tail10_pose_sim3_rmse_m": 0.10,
    "scale_cv_head_mid_tail_pose_sim3": 0.10,
}


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _parse_csv_text(text: str) -> list[str]:
    return [part.strip() for part in str(text or "").split(",") if part.strip()]


def _seq_norm(value: Any) -> str:
    return f"{int(str(value).strip()):02d}"


def _selected_cases_include_dedicated_qk_pair(names: list[str]) -> bool:
    return any(any(marker in str(name) for marker in DEDICATED_QK_PAIR_CASE_MARKERS) for name in names)


def _finite(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


def _median(values: list[float]) -> float | None:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    if not vals:
        return None
    return float(np.median(np.asarray(vals, dtype=float)))


def _ratio_improvement(base: float | None, cand: float | None) -> float | None:
    if base is None or cand is None:
        return None
    denom = max(abs(float(base)), 1e-12)
    return float((float(base) - float(cand)) / denom)


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
            clean: dict[str, Any] = {}
            for key in fieldnames:
                value = row.get(key)
                if isinstance(value, (dict, list, tuple)):
                    clean[key] = json.dumps(_jsonable(value), ensure_ascii=False, sort_keys=True)
                else:
                    clean[key] = value
            writer.writerow(clean)


def _select_targets(args: argparse.Namespace) -> list[dict[str, Any]]:
    seqs = {_seq_norm(seq) for seq in _parse_csv_text(args.seqs)}
    case_types = {str(case_type).strip() for case_type in _parse_csv_text(args.case_types)}
    max_per_bucket = int(getattr(args, "max_targets_per_case_type_per_seq", 0) or 0)
    max_total = int(getattr(args, "max_targets_total", 0) or 0)
    counts: dict[tuple[str, str], int] = {}
    selected: list[dict[str, Any]] = []
    with args.target_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            seq = _seq_norm(row.get("seq") or row.get("sequence"))
            case_type = str(row.get("case_type", "")).strip()
            if seqs and seq not in seqs:
                continue
            if case_types and case_type not in case_types:
                continue
            bucket = (seq, case_type)
            if max_per_bucket > 0 and counts.get(bucket, 0) >= max_per_bucket:
                continue
            target = dict(row)
            target["seq"] = seq
            target["case_type"] = case_type
            target["chunk_id"] = int(row["chunk_id"])
            target["frame_start"] = int(row["frame_start"])
            target["frame_end"] = int(row["frame_end"])
            selected.append(target)
            counts[bucket] = counts.get(bucket, 0) + 1
            if max_total > 0 and len(selected) >= max_total:
                break
    if not selected:
        raise ValueError(f"no targets selected from {args.target_csv}")
    return selected


def _case_key(row: dict[str, Any]) -> tuple[str, str, int]:
    return (str(row["seq"]), str(row["case_type"]), int(row["chunk"]))


def _evaluate_targets(
    *,
    base_dir: Path,
    gt_root: Path,
    targets: list[dict[str, Any]],
    run_names: list[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    gt_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for target in targets:
        seq = str(target["seq"])
        if seq not in gt_cache:
            _, gt_poses_all, gt_pos_all = _load_kitti_gt(gt_root / f"{seq}.txt")
            gt_cache[seq] = (gt_poses_all, gt_pos_all)
        gt_poses_all, gt_pos_all = gt_cache[seq]
        chunk = int(target["chunk_id"])
        case_type = str(target["case_type"])
        for run_name in run_names:
            run_dir = base_dir / f"seq{seq}" / f"chunk{chunk:03d}_{case_type}" / run_name
            try:
                row = _eval_run(run_name, run_dir, gt_poses_all, gt_pos_all, trajectory_name=f"{seq}.txt")
                row.update(
                    {
                        "seq": seq,
                        "case_type": case_type,
                        "chunk": chunk,
                        "phase1_frame_start": int(target["frame_start"]),
                        "phase1_frame_end": int(target["frame_end"]),
                        "phase1_J_short": target.get("J_short"),
                        "phase1_case_rank": target.get("case_rank"),
                    }
                )
                rows.append(row)
            except Exception as exc:  # keep missing/failed runs auditable
                errors.append(
                    {
                        "seq": seq,
                        "case_type": case_type,
                        "chunk": int(chunk),
                        "run": run_name,
                        "run_dir": str(run_dir),
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
    return rows, errors


def _attach_j_proxy(rows: list[dict[str, Any]]) -> None:
    by_case: dict[tuple[str, str, int], dict[str, dict[str, Any]]] = {}
    for row in rows:
        by_case.setdefault(_case_key(row), {})[str(row["run"])] = row
    for case_rows in by_case.values():
        baseline = case_rows.get(BASELINE)
        for row in case_rows.values():
            parts: list[float] = []
            for key in LOWER_IS_BETTER_KEYS:
                base_v = _finite(baseline.get(key)) if baseline is not None else None
                val = _finite(row.get(key))
                if base_v is not None and val is not None:
                    parts.append(float(val) / max(abs(float(base_v)), 1e-12))
            row["J_short_eval_proxy"] = float(np.mean(np.asarray(parts, dtype=float))) if parts else None
            row["J_short_eval_proxy_source"] = "mean(local/head_tail/scale_cv normalized by same-case READ0_NATIVE)"


def _aggregate_by_case_type(
    rows: list[dict[str, Any]],
    *,
    run_name: str,
    case_type: str,
    key: str,
) -> float | None:
    vals = [_finite(row.get(key)) for row in rows if str(row.get("run")) == run_name and str(row.get("case_type")) == case_type]
    return _median([float(v) for v in vals if v is not None])


def _control_detail(
    rows: list[dict[str, Any]],
    *,
    controls: list[str],
    case_type: str,
    key: str,
) -> dict[str, Any]:
    values: dict[str, Any] = {}
    missing: list[str] = []
    for control in controls:
        val = _aggregate_by_case_type(rows, run_name=control, case_type=case_type, key=key)
        if val is None:
            missing.append(control)
        else:
            values[control] = val
    finite_vals = [float(v) for v in values.values()]
    return {
        "median_values": values,
        "missing_or_invalid_controls": missing,
        "best_control_median": float(min(finite_vals)) if finite_vals else None,
        "all_controls_finite": bool(len(values) == len(controls)),
    }


def _candidate_decision(
    rows: list[dict[str, Any]],
    *,
    candidate: str,
    controls: list[str],
) -> dict[str, Any]:
    comparisons: dict[str, Any] = {}
    bad_pass_keys: list[str] = []
    for key in ["J_short_eval_proxy"] + LOWER_IS_BETTER_KEYS:
        bad_base = _aggregate_by_case_type(rows, run_name=BASELINE, case_type="bad", key=key)
        bad_cand = _aggregate_by_case_type(rows, run_name=candidate, case_type="bad", key=key)
        good_base = _aggregate_by_case_type(rows, run_name=BASELINE, case_type="good", key=key)
        good_cand = _aggregate_by_case_type(rows, run_name=candidate, case_type="good", key=key)
        bad_improve = _ratio_improvement(bad_base, bad_cand)
        good_worsen = None
        if good_base is not None and good_cand is not None:
            good_worsen = float((float(good_cand) - float(good_base)) / max(abs(float(good_base)), 1e-12))
        ctrl = _control_detail(rows, controls=controls, case_type="bad", key=key)
        best_ctrl = _finite(ctrl.get("best_control_median"))
        beats_controls = bool(
            bad_cand is not None
            and best_ctrl is not None
            and bool(ctrl["all_controls_finite"])
            and float(bad_cand) < float(best_ctrl)
        )
        threshold = PLAN_THRESHOLD_BY_KEY[key]
        bad_key_pass = bool(bad_improve is not None and bad_improve >= threshold and beats_controls)
        if bad_key_pass:
            bad_pass_keys.append(key)
        comparisons[key] = {
            "bad_baseline_median": bad_base,
            "bad_candidate_median": bad_cand,
            "bad_improvement_vs_baseline_ratio": bad_improve,
            "good_baseline_median": good_base,
            "good_candidate_median": good_cand,
            "good_worsen_ratio": good_worsen,
            "bad_candidate_beats_all_controls": beats_controls,
            "bad_control_detail": ctrl,
            "threshold": threshold,
            "bad_key_pass": bad_key_pass,
        }
    good_j = comparisons["J_short_eval_proxy"].get("good_worsen_ratio")
    good_safety_pass = bool(good_j is not None and good_j <= 0.02)
    phase3_gate_pass = bool(bad_pass_keys and good_safety_pass)
    return {
        "candidate": candidate,
        "baseline": BASELINE,
        "required_controls": controls,
        "bad_metric_passes": bad_pass_keys,
        "good_safety_pass": good_safety_pass,
        "phase3_existing_actuator_gate_pass": phase3_gate_pass,
        "comparisons": comparisons,
        "rule": (
            "Existing-actuator Phase3 pass requires bad cases median J_short_eval_proxy improves >=5% "
            "or local/head_tail/scale_cv improves >=10%, the passing bad metric beats every required "
            "control, and good cases J_short_eval_proxy does not worsen >2%."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, default=DEFAULT_BASE_DIR)
    parser.add_argument("--target-csv", type=Path, default=DEFAULT_TARGET_CSV)
    parser.add_argument("--gt-root", type=Path, default=DEFAULT_GT_ROOT)
    parser.add_argument("--seqs", default="00,01,02,05")
    parser.add_argument("--case-types", default="bad,good")
    parser.add_argument("--max-targets-per-case-type-per-seq", type=int, default=0)
    parser.add_argument("--max-targets-total", type=int, default=0)
    parser.add_argument("--baseline", default=BASELINE)
    parser.add_argument("--candidate", action="append", default=[])
    parser.add_argument("--control", action="append", default=[])
    parser.add_argument("--out-json", type=Path, default=None)
    parser.add_argument("--out-csv", type=Path, default=None)
    args = parser.parse_args()

    if args.baseline != BASELINE:
        raise ValueError(f"this evaluator currently expects baseline={BASELINE}")
    candidates = args.candidate or DEFAULT_CANDIDATES
    controls = args.control or DEFAULT_CONTROLS
    run_names = list(dict.fromkeys([BASELINE] + candidates + controls))
    out_json = args.out_json or args.base_dir / "phase3_short_read_existing_actuator_gate_summary.json"
    out_csv = args.out_csv or args.base_dir / "phase3_short_read_existing_actuator_metrics.csv"

    targets = _select_targets(args)
    rows, errors = _evaluate_targets(base_dir=args.base_dir, gt_root=args.gt_root, targets=targets, run_names=run_names)
    _attach_j_proxy(rows)
    decisions = {
        candidate: _candidate_decision(rows, candidate=candidate, controls=controls)
        for candidate in candidates
    }
    selected_counts: dict[str, int] = {}
    for target in targets:
        key = f"{target['seq']}:{target['case_type']}"
        selected_counts[key] = selected_counts.get(key, 0) + 1
    payload = {
        "base_dir": str(args.base_dir),
        "target_csv": str(args.target_csv),
        "gt_root": str(args.gt_root),
        "selected_target_count": len(targets),
        "selected_counts": selected_counts,
        "metric_row_count": len(rows),
        "evaluation_error_count": len(errors),
        "evaluation_errors": errors,
        "decisions": decisions,
        "phase3_existing_actuator_gate_pass": bool(any(d.get("phase3_existing_actuator_gate_pass") for d in decisions.values())),
        "actual_method_progress": bool(any(d.get("phase3_existing_actuator_gate_pass") for d in decisions.values())),
        "method_gate_claimed": False,
        "qk_pair_actuator_claimed": bool(_selected_cases_include_dedicated_qk_pair(candidates + controls)),
        "qk_pair_actuator_note": (
            "Dedicated QK-pair cases use frame_bias_mode=qk_pair_* in the runtime frame attention hook. "
            "Legacy READ cases remain existing/proxy actuators."
        ),
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(_jsonable(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_csv(out_csv, rows)
    print(
        json.dumps(
            _jsonable(
                {
                    "phase3_existing_actuator_gate_pass": payload["phase3_existing_actuator_gate_pass"],
                    "actual_method_progress": payload["actual_method_progress"],
                    "metric_row_count": len(rows),
                    "evaluation_error_count": len(errors),
                    "candidate_gate_pass": {
                        name: decision.get("phase3_existing_actuator_gate_pass")
                        for name, decision in decisions.items()
                    },
                    "out_json": out_json,
                    "out_csv": out_csv,
                }
            ),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
