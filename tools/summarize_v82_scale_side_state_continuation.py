#!/usr/bin/env python3
"""Summarize v82 scale-side-state continuation after Phase8 No-Go."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_ROOT = Path(
    "results/acl2_v82tf_swa_carrier_semantic_scale_handoff/"
    "phase8b_scale_side_state_continuation"
)
DEFAULT_SMOKE_ROOT = DEFAULT_ROOT / "seq01_v82_scale_state_pre_guard_s098_102"
DEFAULT_PAIR_BANK = Path(
    "results/acl2_v82tf_swa_carrier_semantic_scale_handoff/"
    "phase2_swa_pair_bank_v2/swa_pair_bank_v2.csv"
)
METRICS = [
    "head10_to_tail10_pose_sim3_rmse_m",
    "overlap3_to_future_pose_sim3_rmse_m",
    "local_sim3_ate_rmse_m",
    "scale_cv_head_mid_tail_pose_sim3",
]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.is_file():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            raw = raw.strip()
            if not raw:
                continue
            item = json.loads(raw)
            if isinstance(item, dict):
                rows.append(item)
    return rows


def _float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if out == out and out not in {float("inf"), float("-inf")} else None


def _median(values: list[float | None]) -> float | None:
    clean = [float(v) for v in values if v is not None]
    return float(statistics.median(clean)) if clean else None


def _ratio_improvement(base: float | None, cand: float | None) -> float | None:
    if base is None or cand is None or abs(base) <= 1.0e-12:
        return None
    return float((base - cand) / base)


def _ratio_worsen(base: float | None, cand: float | None) -> float | None:
    if base is None or cand is None or abs(base) <= 1.0e-12:
        return None
    return float((cand - base) / base)


def _counter(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        value = row.get(key)
        counts[str(value) if value not in {None, ""} else "missing"] += 1
    return dict(sorted(counts.items()))


def _chunk_labels(pair_bank: Path) -> dict[int, dict[str, Any]]:
    labels: dict[int, dict[str, Any]] = {}
    for row in _read_csv(pair_bank):
        if row.get("seq") != "01":
            continue
        chunk = int(row["curr_chunk"])
        labels[chunk] = {
            "base_case_type": row.get("base_case_type", ""),
            "case_type": row.get("case_type", ""),
            "quality_type": row.get("quality_type", ""),
            "future_after_overlap": row.get("future_after_overlap", ""),
            "boundary_jump": row.get("boundary_jump", ""),
            "overlap_scale_residual": row.get("overlap_scale_residual", ""),
        }
    return labels


def _rows_by_chunk_run(rows: list[dict[str, str]]) -> dict[tuple[int, str], dict[str, str]]:
    out: dict[tuple[int, str], dict[str, str]] = {}
    for row in rows:
        out[(int(row["chunk"]), row["run"])] = row
    return out


def _trace_summary(smoke_root: Path, *, chunks: list[int], case: str) -> dict[str, Any]:
    trace_rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for chunk in chunks:
        trace_path = smoke_root / f"chunk{chunk:02d}" / case / "merge_state_trace.jsonl"
        rows = _read_jsonl(trace_path)
        if not rows:
            missing.append(str(trace_path))
        for row in rows:
            item = dict(row)
            item["target_chunk"] = int(chunk)
            trace_rows.append(item)
    active = [row for row in trace_rows if bool(row.get("online_scale_state_active"))]
    nonrefresh = [
        row
        for row in active
        if str(row.get("online_scale_state_action") or "").strip().lower() not in {"", "refresh"}
    ]
    gate_rows = [row for row in trace_rows if "online_scale_state_gate_pass" in row]
    guard_rows = [row for row in trace_rows if "semantic_merge_native_overlap_guard_rejected" in row]
    input_scales = [_float(row.get("online_scale_state_input_scale")) for row in trace_rows]
    output_scales = [_float(row.get("online_scale_state_output_scale")) for row in trace_rows]
    transform_scales = [_float(row.get("transform_scale_value")) for row in trace_rows]
    input_scales = [value for value in input_scales if value is not None]
    output_scales = [value for value in output_scales if value is not None]
    transform_scales = [value for value in transform_scales if value is not None]
    return {
        "case": case,
        "trace_rows": len(trace_rows),
        "missing_trace_paths": missing,
        "scale_state_active_rows": len(active),
        "scale_state_nonrefresh_rows": len(nonrefresh),
        "scale_state_action_counts": _counter(active, "online_scale_state_action"),
        "scale_state_stage_counts": _counter(active, "online_scale_state_stage"),
        "scale_state_gate_rows": len(gate_rows),
        "scale_state_gate_pass_counts": _counter(gate_rows, "online_scale_state_gate_pass"),
        "scale_state_gate_reason_counts": _counter(gate_rows, "online_scale_state_gate_reason"),
        "native_overlap_guard_rows": len(guard_rows),
        "native_overlap_guard_reject_counts": _counter(guard_rows, "semantic_merge_native_overlap_guard_rejected"),
        "input_scale_min": min(input_scales) if input_scales else None,
        "input_scale_max": max(input_scales) if input_scales else None,
        "output_scale_min": min(output_scales) if output_scales else None,
        "output_scale_max": max(output_scales) if output_scales else None,
        "transform_scale_min": min(transform_scales) if transform_scales else None,
        "transform_scale_max": max(transform_scales) if transform_scales else None,
    }


def _candidate_summary(
    *,
    candidate: str,
    rows_csv: Path,
    baseline: str,
    controls: list[str],
    chunk_labels: dict[int, dict[str, Any]],
    evaluator_summary: Path,
    smoke_root: Path,
) -> dict[str, Any]:
    rows = _rows_by_chunk_run(_read_csv(rows_csv))
    chunks = sorted(chunk_labels)
    per_chunk: list[dict[str, Any]] = []
    for chunk in chunks:
        label = chunk_labels[chunk]
        base = rows.get((chunk, baseline), {})
        cand = rows.get((chunk, candidate), {})
        control_rows = [rows.get((chunk, control), {}) for control in controls]
        item: dict[str, Any] = {
            "chunk": chunk,
            "base_case_type": label["base_case_type"],
            "case_type": label["case_type"],
            "quality_type": label["quality_type"],
        }
        for metric in METRICS:
            base_v = _float(base.get(metric))
            cand_v = _float(cand.get(metric))
            control_values = [_float(row.get(metric)) for row in control_rows]
            control_values = [value for value in control_values if value is not None]
            best_control = min(control_values) if control_values else None
            item[f"{metric}_baseline"] = base_v
            item[f"{metric}_candidate"] = cand_v
            item[f"{metric}_best_control"] = best_control
            item[f"{metric}_improvement_vs_baseline_ratio"] = _ratio_improvement(base_v, cand_v)
            item[f"{metric}_worsen_vs_baseline_ratio"] = _ratio_worsen(base_v, cand_v)
            item[f"{metric}_beats_controls"] = bool(
                cand_v is not None and best_control is not None and cand_v < best_control
            )
        per_chunk.append(item)

    bad = [row for row in per_chunk if row["base_case_type"] == "bad"]
    good = [row for row in per_chunk if row["base_case_type"] == "good"]
    metric_summary: dict[str, Any] = {}
    for metric in METRICS:
        bad_improvements = [row.get(f"{metric}_improvement_vs_baseline_ratio") for row in bad]
        good_worsens = [row.get(f"{metric}_worsen_vs_baseline_ratio") for row in good]
        good_worsens_clean = [value for value in good_worsens if value is not None]
        metric_summary[metric] = {
            "bad_median_improvement_vs_baseline_ratio": _median(bad_improvements),
            "bad_control_beat_count": sum(1 for row in bad if row.get(f"{metric}_beats_controls")),
            "good_max_worsen_vs_baseline_ratio": max(good_worsens_clean) if good_worsens_clean else None,
            "good_worsen_le_2pct_all": bool(good_worsens_clean and max(good_worsens_clean) <= 0.02),
        }

    eval_payload = _read_json(evaluator_summary)
    trace = _trace_summary(smoke_root, chunks=chunks, case=candidate)
    head = metric_summary["head10_to_tail10_pose_sim3_rmse_m"]
    future = metric_summary["overlap3_to_future_pose_sim3_rmse_m"]
    good_ok = all(item["good_worsen_le_2pct_all"] for item in metric_summary.values())
    bad_proxy_pass = bool(
        (head["bad_median_improvement_vs_baseline_ratio"] or 0.0) >= 0.05
        or (future["bad_median_improvement_vs_baseline_ratio"] or 0.0) >= 0.10
    )
    control_ok = bool(
        head["bad_control_beat_count"] >= len(bad)
        or future["bad_control_beat_count"] >= len(bad)
    )
    gate = {
        "bad_rows_eq_3": len(bad) == 3,
        "good_rows_eq_3": len(good) == 3,
        "candidate_scale_state_active_rows_gt0": trace["scale_state_active_rows"] > 0,
        "candidate_scale_state_nonrefresh_rows_gt0": trace["scale_state_nonrefresh_rows"] > 0,
        "bad_jmid_proxy_or_future_ge_threshold": bad_proxy_pass,
        "good_worsen_le_2pct_all_metrics": good_ok,
        "bad_beats_controls_all_for_a_proxy_metric": control_ok,
        "evaluator_phaseE_gate_pass": bool(eval_payload.get("phaseE_gate_pass")),
    }
    gate["candidate_phase8b_gate_pass"] = all(gate.values())
    return {
        "candidate": candidate,
        "baseline": baseline,
        "controls": controls,
        "chunks": chunks,
        "bad_chunks": [row["chunk"] for row in bad],
        "good_chunks": [row["chunk"] for row in good],
        "metric_summary": metric_summary,
        "trace_summary": trace,
        "control_trace_summary": {
            control: _trace_summary(smoke_root, chunks=chunks, case=control) for control in controls
        },
        "gate": gate,
        "per_chunk": per_chunk,
        "evaluator_summary": str(evaluator_summary),
        "evaluator_phaseE_gate_pass": bool(eval_payload.get("phaseE_gate_pass")),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--smoke-root", type=Path, default=DEFAULT_SMOKE_ROOT)
    parser.add_argument("--pair-bank", type=Path, default=DEFAULT_PAIR_BANK)
    parser.add_argument(
        "--prefix",
        default="phase8b",
        help="Evaluator output filename prefix inside --smoke-root.",
    )
    args = parser.parse_args()

    labels = _chunk_labels(args.pair_bank)
    prefix = str(args.prefix or "phase8b").strip()
    candidates = [
        _candidate_summary(
            candidate="overlap_outlier",
            rows_csv=args.smoke_root / f"{prefix}_overlap_outlier_run_metrics.csv",
            baseline="native_no_swa",
            controls=["geometry_only", "overlap_outlier_random", "overlap_outlier_shuffled"],
            chunk_labels=labels,
            evaluator_summary=args.smoke_root / f"{prefix}_overlap_outlier_gate_summary.json",
            smoke_root=args.smoke_root,
        ),
        _candidate_summary(
            candidate="robust_semoverlap",
            rows_csv=args.smoke_root / f"{prefix}_robust_semoverlap_run_metrics.csv",
            baseline="native_no_swa",
            controls=["geometry_only", "robust_semoverlap_random", "robust_semoverlap_shuffled"],
            chunk_labels=labels,
            evaluator_summary=args.smoke_root / f"{prefix}_robust_semoverlap_gate_summary.json",
            smoke_root=args.smoke_root,
        ),
    ]
    passing = [item["candidate"] for item in candidates if item["gate"]["candidate_phase8b_gate_pass"]]
    summary = {
        "schema": "acl2_v82_phase8b_scale_side_state_continuation_summary_v1",
        "smoke_root": str(args.smoke_root),
        "evaluator_prefix": prefix,
        "candidate_count": len(candidates),
        "passing_candidates": passing,
        "phase8b_gate_pass": bool(passing),
        "decision": (
            "pass_to_phase9_ttt_after_scale_side_state_confirmation"
            if passing
            else "no_go_scale_side_state_failed_before_ttt"
        ),
        "blocker": (
            ""
            if passing
            else "No explicit scale-side-state candidate met action, bad-pair improvement, good-pair protection, and control-beating gates."
        ),
        "rule": (
            "Continuation after Phase8 No-Go. Pass requires candidate scale-state action to occur, "
            "bad seq01 chunks to improve by >=5% head-tail or >=10% overlap-future, all good chunks "
            "to worsen <=2%, and the candidate to beat geometry/random/shuffled controls."
        ),
        "candidates": candidates,
    }
    args.root.mkdir(parents=True, exist_ok=True)
    out_json = args.root / "scale_side_state_continuation_summary.json"
    out_report = args.root / "scale_side_state_continuation_report.md"
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# v82 Phase8b Scale-Side-State Continuation Summary",
        "",
        f"phase8b_gate_pass: {summary['phase8b_gate_pass']}",
        f"decision: {summary['decision']}",
        f"blocker: {summary['blocker'] or 'none'}",
        "",
        summary["rule"],
        "",
    ]
    for item in candidates:
        lines.extend(
            [
                f"## {item['candidate']}",
                "",
                f"evaluator_phaseE_gate_pass: {item['evaluator_phaseE_gate_pass']}",
                f"candidate_phase8b_gate_pass: {item['gate']['candidate_phase8b_gate_pass']}",
                f"bad_chunks: {item['bad_chunks']}",
                f"good_chunks: {item['good_chunks']}",
                f"gate: {item['gate']}",
                f"trace_summary: {item['trace_summary']}",
                f"metric_summary: {item['metric_summary']}",
                "",
            ]
        )
    out_report.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "candidates"}, ensure_ascii=False, indent=2, sort_keys=True))
    print(f"wrote_json={out_json}")
    print(f"wrote_report={out_report}")


if __name__ == "__main__":
    main()
