#!/usr/bin/env python3
"""Mechanism-depth analysis for ACL2 v80 Phase3 short READ metrics."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_METRICS_CSV = Path(
    "results/kitti01_hmc_v2/acl2_v80tf_multiseq_goodbad_semantic_three_memory_control/"
    "report_final/phase3_short_read_existing_actuator_control/rollouts/"
    "phase3_short_read_existing_actuator_metrics.csv"
)
DEFAULT_GATE_JSON = Path(
    "results/kitti01_hmc_v2/acl2_v80tf_multiseq_goodbad_semantic_three_memory_control/"
    "report_final/phase3_short_read_existing_actuator_control/rollouts/"
    "phase3_short_read_existing_actuator_gate_summary.json"
)
DEFAULT_OUT_JSON = Path(
    "results/kitti01_hmc_v2/acl2_v80tf_multiseq_goodbad_semantic_three_memory_control/"
    "report_final/phase3_short_read_existing_actuator_control/rollouts/"
    "phase3_short_read_mechanism_depth_analysis.json"
)
DEFAULT_OUT_CSV = Path(
    "results/kitti01_hmc_v2/acl2_v80tf_multiseq_goodbad_semantic_three_memory_control/"
    "report_final/phase3_short_read_existing_actuator_control/rollouts/"
    "phase3_short_read_case_delta_rank.csv"
)

BASELINE = "READ0_NATIVE"
DEFAULT_CANDIDATE = "READ1_EXISTING_L07_LAYOUT_SELECT"
DEFAULT_CONTROLS = [
    "READ7_GEOMETRY_ONLY_CONTROL",
    "READ8_LABEL_SHUFFLE",
    "READ9_CONFIDENCE_SHUFFLE",
    "READ10_SAME_READ_MASS_RANDOM",
    "READ11_GROUP_STRATIFIED_RANDOM",
]
METRICS = [
    "J_short_eval_proxy",
    "local_sim3_ate_rmse_m",
    "head10_to_tail10_pose_sim3_rmse_m",
    "scale_cv_head_mid_tail_pose_sim3",
]
HOOK_FIELDS = [
    "frame_attention_num_calls",
    "frame_attention_num_enabled_layers",
    "frame_attention_mean_abs_bias",
    "frame_attention_max_abs_bias",
    "prior_v78_l07_l13_available",
    "prior_v78_l07_l13_output_mean",
    "prior_v78_l07_l13_output_gt050_mass",
    "prior_v78_l07_l13_control_effective",
    "prior_v78_l07_l13_control_shuffle_mode",
]


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


def _finite(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


def _boolish(value: Any) -> bool | None:
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    return None


def _median(values: list[float]) -> float | None:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    if not vals:
        return None
    return float(np.median(np.asarray(vals, dtype=float)))


def _mean(values: list[float]) -> float | None:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    if not vals:
        return None
    return float(np.mean(np.asarray(vals, dtype=float)))


def _ratio_improvement(base: float | None, value: float | None) -> float | None:
    if base is None or value is None:
        return None
    return float((float(base) - float(value)) / max(abs(float(base)), 1e-12))


def _load_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _case_id(row: dict[str, Any]) -> tuple[str, str, int]:
    return (str(row.get("seq")), str(row.get("case_type")), int(float(row.get("chunk"))))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _hook_summary(rows: list[dict[str, Any]], run_names: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for run in run_names:
        group = [row for row in rows if str(row.get("run")) == run]
        item: dict[str, Any] = {"row_count": len(group)}
        for field in HOOK_FIELDS:
            vals = [_finite(row.get(field)) for row in group]
            vals_f = [float(v) for v in vals if v is not None]
            bools = [_boolish(row.get(field)) for row in group]
            bools_f = [v for v in bools if v is not None]
            if vals_f:
                item[f"{field}_median"] = _median(vals_f)
                item[f"{field}_mean"] = _mean(vals_f)
                item[f"{field}_max"] = max(vals_f)
            elif bools_f:
                item[f"{field}_true_frac"] = float(sum(1 for v in bools_f if v) / max(len(bools_f), 1))
            else:
                counts = Counter(str(row.get(field)) for row in group if str(row.get(field, "")).strip())
                if counts:
                    item[f"{field}_counts"] = dict(counts)
        out[run] = item
    return out


def _case_delta_rows(
    rows: list[dict[str, Any]],
    *,
    candidate: str,
    controls: list[str],
) -> list[dict[str, Any]]:
    by_case: dict[tuple[str, str, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        by_case[_case_id(row)][str(row.get("run"))] = row
    out: list[dict[str, Any]] = []
    for (seq, case_type, chunk), case_rows in sorted(by_case.items()):
        base = case_rows.get(BASELINE)
        cand = case_rows.get(candidate)
        if base is None or cand is None:
            out.append(
                {
                    "seq": seq,
                    "case_type": case_type,
                    "chunk": chunk,
                    "status": "missing_baseline_or_candidate",
                    "available_runs": ",".join(sorted(case_rows)),
                }
            )
            continue
        row_out: dict[str, Any] = {
            "seq": seq,
            "case_type": case_type,
            "chunk": chunk,
            "status": "ok",
            "candidate": candidate,
        }
        best_control_by_metric: dict[str, str] = {}
        for metric in METRICS:
            base_v = _finite(base.get(metric))
            cand_v = _finite(cand.get(metric))
            row_out[f"{metric}_baseline"] = base_v
            row_out[f"{metric}_candidate"] = cand_v
            row_out[f"{metric}_candidate_improvement_ratio"] = _ratio_improvement(base_v, cand_v)
            control_values: list[tuple[str, float]] = []
            for control in controls:
                ctrl_row = case_rows.get(control)
                ctrl_v = _finite(ctrl_row.get(metric)) if ctrl_row is not None else None
                row_out[f"{metric}_{control}"] = ctrl_v
                if ctrl_v is not None:
                    control_values.append((control, float(ctrl_v)))
            if control_values:
                best_control, best_value = min(control_values, key=lambda item: item[1])
                row_out[f"{metric}_best_control"] = best_control
                row_out[f"{metric}_best_control_value"] = best_value
                row_out[f"{metric}_candidate_minus_best_control"] = (
                    float(cand_v) - float(best_value) if cand_v is not None else None
                )
                best_control_by_metric[metric] = best_control
        row_out["candidate_hook_mean_abs_bias"] = _finite(cand.get("frame_attention_mean_abs_bias"))
        row_out["candidate_hook_max_abs_bias"] = _finite(cand.get("frame_attention_max_abs_bias"))
        row_out["candidate_v78_available"] = cand.get("prior_v78_l07_l13_available")
        row_out["candidate_v78_output_gt050_mass"] = _finite(cand.get("prior_v78_l07_l13_output_gt050_mass"))
        row_out["best_control_votes"] = json.dumps(best_control_by_metric, sort_keys=True)
        out.append(row_out)
    return out


def _classify(payload: dict[str, Any], rows: list[dict[str, Any]], case_delta_rows: list[dict[str, Any]], candidate: str) -> dict[str, Any]:
    cand_rows = [row for row in rows if str(row.get("run")) == candidate]
    hook_mean = _median([v for v in (_finite(row.get("frame_attention_mean_abs_bias")) for row in cand_rows) if v is not None])
    hook_max = _median([v for v in (_finite(row.get("frame_attention_max_abs_bias")) for row in cand_rows) if v is not None])
    v78_available_frac_vals = [_boolish(row.get("prior_v78_l07_l13_available")) for row in cand_rows]
    v78_available_frac = (
        sum(1 for v in v78_available_frac_vals if v) / max(sum(1 for v in v78_available_frac_vals if v is not None), 1)
        if any(v is not None for v in v78_available_frac_vals)
        else None
    )
    bad_j_improvements = [
        _finite(row.get("J_short_eval_proxy_candidate_improvement_ratio"))
        for row in case_delta_rows
        if row.get("case_type") == "bad"
    ]
    bad_j_median_improvement = _median([float(v) for v in bad_j_improvements if v is not None])
    best_control_counts = Counter()
    for row in case_delta_rows:
        if row.get("case_type") != "bad":
            continue
        try:
            votes = json.loads(str(row.get("best_control_votes") or "{}"))
        except json.JSONDecodeError:
            votes = {}
        best_control_counts.update(str(v) for v in votes.values())
    gate_pass = bool(payload.get("phase3_existing_actuator_gate_pass") or payload.get("actual_method_progress"))
    if gate_pass:
        class_name = "phase3_existing_actuator_success"
    elif hook_mean is None or hook_max is None or (hook_mean <= 1e-9 and hook_max <= 1e-9):
        class_name = "hook_inactive_or_zero_bias"
    elif v78_available_frac is not None and v78_available_frac < 0.99:
        class_name = "semantic_cue_unavailable_on_some_cases"
    elif bad_j_median_improvement is None or bad_j_median_improvement <= 0.0:
        class_name = "actuator_active_but_no_bad_case_metric_improvement"
    elif best_control_counts and best_control_counts.most_common(1)[0][1] >= 3:
        class_name = "semantic_not_specific_controls_explain_or_beat_candidate"
    else:
        class_name = "partial_response_but_gate_failed"
    return {
        "failure_or_success_class": class_name,
        "candidate_hook_mean_abs_bias_median": hook_mean,
        "candidate_hook_max_abs_bias_median": hook_max,
        "candidate_v78_available_frac": v78_available_frac,
        "bad_J_short_eval_proxy_median_improvement": bad_j_median_improvement,
        "bad_best_control_vote_counts": dict(best_control_counts),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics-csv", type=Path, default=DEFAULT_METRICS_CSV)
    parser.add_argument("--gate-json", type=Path, default=DEFAULT_GATE_JSON)
    parser.add_argument("--candidate", default=DEFAULT_CANDIDATE)
    parser.add_argument("--controls", default=",".join(DEFAULT_CONTROLS))
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out-csv", type=Path, default=DEFAULT_OUT_CSV)
    args = parser.parse_args()

    controls = [part.strip() for part in str(args.controls).split(",") if part.strip()]
    rows = _load_rows(args.metrics_csv)
    payload = json.loads(args.gate_json.read_text(encoding="utf-8")) if args.gate_json.exists() else {}
    run_names = list(dict.fromkeys([BASELINE, args.candidate] + controls))
    deltas = _case_delta_rows(rows, candidate=args.candidate, controls=controls)
    analysis = {
        "metrics_csv": str(args.metrics_csv),
        "gate_json": str(args.gate_json),
        "row_count": len(rows),
        "case_delta_row_count": len(deltas),
        "candidate": args.candidate,
        "controls": controls,
        "gate": {
            "phase3_existing_actuator_gate_pass": bool(payload.get("phase3_existing_actuator_gate_pass")),
            "actual_method_progress": bool(payload.get("actual_method_progress")),
            "evaluation_error_count": payload.get("evaluation_error_count"),
            "metric_row_count": payload.get("metric_row_count"),
        },
        "hook_summary_by_run": _hook_summary(rows, run_names),
    }
    analysis.update(_classify(payload, rows, deltas, args.candidate))
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(_jsonable(analysis), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_csv(args.out_csv, deltas)
    print(json.dumps(_jsonable(analysis), ensure_ascii=False, indent=2, sort_keys=True))
    print(f"wrote_json={args.out_json}")
    print(f"wrote_csv={args.out_csv}")


if __name__ == "__main__":
    main()
