#!/usr/bin/env python3
"""Audit whether existing v82 Phase8e merge actions contain a viable controller.

This is diagnostic only. GT oracles are explicitly marked non-runtime and must
not be promoted as methods.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any, Mapping


ROOT = Path("results/acl2_v82tf_swa_carrier_semantic_scale_handoff")
DEFAULT_OUT = ROOT / "phase12_merge_controller_oracle_audit"
DEFAULT_PAIR_BANK = ROOT / "phase2_swa_pair_bank_v2/swa_pair_bank_v2.csv"
DEFAULT_PHASE8E = ROOT / "phase8e_projection_tol001_steps64_continuation"
DEFAULT_PHASE8E_SMOKE = DEFAULT_PHASE8E / "seq01_v82_projection_tol001_steps64"
CHUNKS = [2, 6, 9, 11, 14, 17]
BAD_CHUNKS = [9, 11, 14]
GOOD_CHUNKS = [2, 6, 17]
SEMANTIC_RUNS = ["overlap_outlier", "robust_semoverlap"]
ALL_RUNS = [
    "native_no_swa",
    "geometry_only",
    "overlap_outlier",
    "overlap_outlier_random",
    "overlap_outlier_shuffled",
    "robust_semoverlap",
    "robust_semoverlap_random",
    "robust_semoverlap_shuffled",
]
CONTROL_RUNS = [
    "geometry_only",
    "overlap_outlier_random",
    "overlap_outlier_shuffled",
    "robust_semoverlap_random",
    "robust_semoverlap_shuffled",
]
METRICS = {
    "head_tail": "head10_to_tail10_pose_sim3_rmse_m",
    "overlap_future": "overlap3_to_future_pose_sim3_rmse_m",
}


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _write_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, ensure_ascii=False, sort_keys=True)
                    if isinstance(value, (dict, list, tuple))
                    else value
                    for key, value in ((key, row.get(key, "")) for key in fields)
                }
            )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.is_file():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            raw = raw.strip()
            if not raw:
                continue
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def _float(value: Any) -> float | None:
    try:
        text = str(value).strip()
        if not text:
            return None
        value = float(text)
    except (TypeError, ValueError):
        return None
    return value if value == value else None


def _ratio(base: float | None, cand: float | None) -> float | None:
    if base is None or cand is None or abs(base) <= 1.0e-12:
        return None
    return float((base - cand) / base)


def _median(values: list[float]) -> float | None:
    return float(statistics.median(values)) if values else None


def _rows_by_chunk_run(phase8e_root: Path) -> dict[tuple[int, str], dict[str, str]]:
    rows: dict[tuple[int, str], dict[str, str]] = {}
    for path in [
        phase8e_root / "phase8e_overlap_outlier_run_metrics.csv",
        phase8e_root / "phase8e_robust_semoverlap_run_metrics.csv",
    ]:
        for row in _read_csv(path):
            rows[(int(float(row["chunk"])), row["run"])] = row
    return rows


def _trace(phase8e_smoke: Path, chunk: int, run: str) -> dict[str, Any]:
    rows = _read_jsonl(phase8e_smoke / f"chunk{chunk:02d}" / run / "merge_state_trace.jsonl")
    return next((row for row in rows if row.get("local_chunk_idx") == 1), rows[-1] if rows else {})


def _metric(rows: Mapping[tuple[int, str], Mapping[str, str]], chunk: int, run: str, metric: str) -> float | None:
    row = rows.get((chunk, run), {})
    return _float(row.get(metric))


def _best_run(
    rows: Mapping[tuple[int, str], Mapping[str, str]],
    chunk: int,
    runs: list[str],
    metric: str,
) -> str:
    best = "native_no_swa"
    best_value = _metric(rows, chunk, best, metric)
    for run in runs:
        value = _metric(rows, chunk, run, metric)
        if value is not None and (best_value is None or value < best_value):
            best = run
            best_value = value
    return best


def _runtime_native_overlap_proxy(
    rows: Mapping[tuple[int, str], Mapping[str, str]],
    phase8e_smoke: Path,
    chunk: int,
) -> str:
    best_run = "native_no_swa"
    best_gain = 0.0
    for run in SEMANTIC_RUNS:
        trace = _trace(phase8e_smoke, chunk, run)
        native = _float(trace.get("semantic_merge_native_overlap_residual"))
        final = _float(trace.get("semantic_merge_final_overlap_residual"))
        if native is None or final is None:
            continue
        gain = native - final
        if gain > best_gain:
            best_gain = gain
            best_run = run
    return best_run


def _scale_delta_proxy(
    rows: Mapping[tuple[int, str], Mapping[str, str]],
    phase8e_smoke: Path,
    chunk: int,
) -> str:
    best_run = "native_no_swa"
    best_score = 0.0
    for run in SEMANTIC_RUNS:
        trace = _trace(phase8e_smoke, chunk, run)
        scale = _float(trace.get("transform_scale_value"))
        native = _float(trace.get("semantic_merge_native_overlap_residual"))
        final = _float(trace.get("semantic_merge_final_overlap_residual"))
        if scale is None or native is None or final is None:
            continue
        if final > native + 0.001:
            continue
        score = abs(scale - 1.0)
        if score >= 0.015 and score > best_score:
            best_score = score
            best_run = run
    return best_run


def _selection_for(controller: str, rows: Mapping[tuple[int, str], Mapping[str, str]], phase8e_smoke: Path) -> dict[int, str]:
    out: dict[int, str] = {}
    for chunk in CHUNKS:
        if controller == "semantic_gt_head_oracle":
            out[chunk] = _best_run(rows, chunk, ["native_no_swa", *SEMANTIC_RUNS], METRICS["head_tail"])
        elif controller == "semantic_gt_overlap_oracle":
            out[chunk] = _best_run(rows, chunk, ["native_no_swa", *SEMANTIC_RUNS], METRICS["overlap_future"])
        elif controller == "runtime_native_overlap_proxy":
            out[chunk] = _runtime_native_overlap_proxy(rows, phase8e_smoke, chunk)
        elif controller == "scale_delta_proxy":
            out[chunk] = _scale_delta_proxy(rows, phase8e_smoke, chunk)
        elif controller == "all_run_gt_upper_bound":
            out[chunk] = _best_run(rows, chunk, ALL_RUNS, METRICS["head_tail"])
        else:
            raise ValueError(controller)
    return out


def _evaluate_controller(
    controller: str,
    selection: Mapping[int, str],
    rows: Mapping[tuple[int, str], Mapping[str, str]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    invalid = controller in {"semantic_gt_head_oracle", "semantic_gt_overlap_oracle", "all_run_gt_upper_bound"}
    per_chunk: list[dict[str, Any]] = []
    metric_summary: dict[str, Any] = {}
    for chunk, run in selection.items():
        row: dict[str, Any] = {"controller": controller, "chunk": chunk, "selected_run": run}
        for name, metric in METRICS.items():
            base = _metric(rows, chunk, "native_no_swa", metric)
            cand = _metric(rows, chunk, run, metric)
            controls = [_metric(rows, chunk, control, metric) for control in CONTROL_RUNS]
            controls = [value for value in controls if value is not None]
            best_control = min(controls) if controls else None
            row[f"{name}_baseline"] = base
            row[f"{name}_selected"] = cand
            row[f"{name}_best_control"] = best_control
            row[f"{name}_improvement_vs_baseline_ratio"] = _ratio(base, cand)
            row[f"{name}_beats_controls"] = cand is not None and best_control is not None and cand < best_control
        per_chunk.append(row)

    for name in METRICS:
        bad_improvements = [
            row[f"{name}_improvement_vs_baseline_ratio"]
            for row in per_chunk
            if row["chunk"] in BAD_CHUNKS and row.get(f"{name}_improvement_vs_baseline_ratio") is not None
        ]
        good_worsens = [
            -row[f"{name}_improvement_vs_baseline_ratio"]
            for row in per_chunk
            if row["chunk"] in GOOD_CHUNKS and row.get(f"{name}_improvement_vs_baseline_ratio") is not None
        ]
        bad_control_beats = sum(
            1 for row in per_chunk if row["chunk"] in BAD_CHUNKS and row.get(f"{name}_beats_controls")
        )
        threshold = 0.05 if name == "head_tail" else 0.10
        metric_summary[name] = {
            "bad_median_improvement_vs_baseline_ratio": _median([x for x in bad_improvements if x is not None]),
            "good_max_worsen_vs_baseline_ratio": max(good_worsens) if good_worsens else None,
            "bad_control_beat_count": bad_control_beats,
            "metric_gate_pass": bool(
                _median([x for x in bad_improvements if x is not None]) is not None
                and _median([x for x in bad_improvements if x is not None]) >= threshold
                and good_worsens
                and max(good_worsens) <= 0.02
                and bad_control_beats >= len(BAD_CHUNKS)
                and not invalid
            ),
            "threshold": threshold,
        }
    summary = {
        "controller": controller,
        "invalid_as_runtime_method": invalid,
        "invalid_reason": "uses GT/evaluator metrics or controls for selection" if invalid else "",
        "selection": dict(selection),
        "metric_summary": metric_summary,
        "controller_gate_pass": bool(any(item["metric_gate_pass"] for item in metric_summary.values())),
    }
    return summary, per_chunk


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--pair-bank", type=Path, default=DEFAULT_PAIR_BANK)
    parser.add_argument("--phase8e-root", type=Path, default=DEFAULT_PHASE8E)
    parser.add_argument("--phase8e-smoke-root", type=Path, default=DEFAULT_PHASE8E_SMOKE)
    args = parser.parse_args()
    args.out_root.mkdir(parents=True, exist_ok=True)

    rows = _rows_by_chunk_run(args.phase8e_root)
    controllers = [
        "semantic_gt_head_oracle",
        "semantic_gt_overlap_oracle",
        "runtime_native_overlap_proxy",
        "scale_delta_proxy",
        "all_run_gt_upper_bound",
    ]
    summaries: list[dict[str, Any]] = []
    per_chunk_rows: list[dict[str, Any]] = []
    for controller in controllers:
        selection = _selection_for(controller, rows, args.phase8e_smoke_root)
        summary, per_chunk = _evaluate_controller(controller, selection, rows)
        summaries.append(summary)
        per_chunk_rows.extend(per_chunk)

    out = {
        "schema": "acl2_v82_phase12_merge_controller_oracle_audit_v1",
        "phase8e_root": str(args.phase8e_root),
        "chunks": CHUNKS,
        "bad_chunks": BAD_CHUNKS,
        "good_chunks": GOOD_CHUNKS,
        "controller_count": len(summaries),
        "passing_runtime_controllers": [
            item["controller"] for item in summaries if item["controller_gate_pass"] and not item["invalid_as_runtime_method"]
        ],
        "diagnostic_gt_oracle_note": "GT oracles are upper-bound diagnostics only and cannot be used as runtime policies.",
        "controllers": summaries,
    }
    (args.out_root / "merge_controller_oracle_summary.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_csv(args.out_root / "merge_controller_oracle_per_chunk.csv", per_chunk_rows)
    report = [
        "# v82 Phase12 Merge Controller Oracle Audit",
        "",
        f"passing_runtime_controllers: {out['passing_runtime_controllers']}",
        "",
    ]
    for item in summaries:
        report.extend(
            [
                f"## {item['controller']}",
                "",
                f"invalid_as_runtime_method: {item['invalid_as_runtime_method']}",
                f"controller_gate_pass: {item['controller_gate_pass']}",
                f"selection: {item['selection']}",
                f"metric_summary: {item['metric_summary']}",
                "",
            ]
        )
    (args.out_root / "merge_controller_oracle_report.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "controllers"}, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
