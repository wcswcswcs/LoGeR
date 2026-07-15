#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


RESULTS_ROOT = Path("results/acl2_v114tf_semantic_internal_evidence_quality_memory_influence_control")
ATE_THRESHOLD = 0.05
ROLLING_THRESHOLD = 0.05
MAX_HARM_THRESHOLD = 0.01

RUNS = [
    ("candidate", "HS_LQ1", "stage_hs_lq1_stable_rowmean_neutral_internal_confident_rerun1"),
    ("candidate", "HS_LQ2", "stage_hs_lq2_risk_suppress_internal_mismatch_rerun1"),
    ("candidate", "HS_LQ3", "stage_hs_lq3_stable_plus_risk_internal_quality_rerun1"),
    ("repair", "HS_LQ1_lowgain", "stage_hs_lq1_lowgain_stable_rowmean_neutral_internal_confident_rerun2"),
    ("repair", "HS_LQ3_lowgain", "stage_hs_lq3_lowgain_stable_plus_risk_internal_quality_rerun2"),
    ("repair", "HS_LQ5_headwise_MRT_risk_gate", "stage_hs_lq5_headwise_mrt_risk_gate_mild_rerun1"),
    ("repair", "HS_LQ5_headwise_MRT_delta_clip", "stage_hs_lq5_headwise_mrt_delta_clip_rerun2"),
    ("control", "internal_only", "stage_hs_lq_ctrl_internal_confident_only_rerun1"),
    ("control", "semantic_only", "stage_hs_lq_ctrl_semantic_only_rowmean_neutral_rerun1"),
    ("control", "rowmean_only", "stage_hs_lq_ctrl_rowmean_only_generic_scale_rerun1"),
    ("repair", "HS_LQ3_MRT_risk_gate", "stage_hs_lq3_mrt_risk_gate_mild_rerun1"),
    ("control", "rowmean_MRT_risk_gate_pilot", "stage_hs_lq_ctrl_rowmean_only_generic_scale_mrt_risk_gate_mild_rerun1"),
    (
        "control_full",
        "rowmean_MRT_risk_gate_full_00010205",
        "stage_hs_lq_ctrl_rowmean_only_generic_scale_mrt_risk_gate_mild_rerun1_full_00010205",
    ),
]


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def finite(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if out != out or out in {float("inf"), float("-inf")}:
        return None
    return out


def strict_pass(aggregate: dict[str, Any]) -> bool:
    median_ate = finite(aggregate.get("median_full_ATE_rel_improvement"))
    median_roll = finite(aggregate.get("median_rolling_p90_rel_improvement"))
    max_harm = finite(aggregate.get("max_full_ATE_harm_rel"))
    segment_ok = bool(aggregate.get("segment_scale_not_worse_all"))
    if median_ate is None or median_roll is None or max_harm is None:
        return False
    return bool(
        (median_ate >= ATE_THRESHOLD or median_roll >= ROLLING_THRESHOLD)
        and max_harm <= MAX_HARM_THRESHOLD
        and segment_ok
    )


def build_row(kind: str, label: str, prefix: str, diagnostics: Path) -> dict[str, Any]:
    metric_summary = read_json(diagnostics / f"{prefix}_summary.json")
    aggregate = metric_summary.get("aggregate", {})
    segment_summary = read_json(diagnostics / f"{prefix}_segment_tradeoff_summary.json")
    gate_summary = read_json(diagnostics / f"{prefix}_gate_audit_summary.json").get("aggregate", {})
    mrt_trace_summary = read_json(diagnostics / f"{prefix}_mrt_trace_summary.json").get("aggregate", {})
    comparison_rows = read_csv_rows(diagnostics / f"{prefix}_comparison_rows.csv")
    row: dict[str, Any] = {
        "kind": kind,
        "label": label,
        "prefix": prefix,
        "candidate_name": aggregate.get("candidate_name"),
        "median_full_ATE_rel_improvement": aggregate.get("median_full_ATE_rel_improvement"),
        "median_rolling_p90_rel_improvement": aggregate.get("median_rolling_p90_rel_improvement"),
        "max_full_ATE_harm_rel": aggregate.get("max_full_ATE_harm_rel"),
        "median_segment_scale_rel_improvement": aggregate.get("median_segment_scale_rel_improvement"),
        "improved_seq_count_full_ATE": aggregate.get("improved_seq_count_full_ATE"),
        "segment_scale_not_worse_all": aggregate.get("segment_scale_not_worse_all"),
        "v114_strict_pilot_pass": strict_pass(aggregate),
        "summary_thresholds": {
            "median_full_ATE_or_rolling_ge": ATE_THRESHOLD,
            "max_full_ATE_harm_rel_le": MAX_HARM_THRESHOLD,
            "segment_scale_not_worse_all": True,
        },
        "segment_worse_count": segment_summary.get("worse_segment_count"),
        "segment_better_count": segment_summary.get("better_segment_count"),
        "median_segment_scale_abs_delta_candidate_minus_baseline": segment_summary.get(
            "median_segment_scale_abs_delta_candidate_minus_baseline"
        ),
        "median_segment_ate_rel_improvement": segment_summary.get("median_segment_ate_rel_improvement"),
        "gate_row_count": gate_summary.get("row_count"),
        "gate_std_median": gate_summary.get("gate_std_median"),
        "gate_row_mean_mean_median": gate_summary.get("gate_row_mean_mean_median"),
        "gate_row_mean_std_max": gate_summary.get("gate_row_mean_std_max"),
        "mrt_trace_row_count": mrt_trace_summary.get("row_count"),
        "predicted_metric_scale_delta_count": mrt_trace_summary.get("predicted_metric_scale_delta_count"),
        "predicted_metric_scale_delta_min": mrt_trace_summary.get("predicted_metric_scale_delta_min"),
        "predicted_metric_scale_delta_max": mrt_trace_summary.get("predicted_metric_scale_delta_max"),
        "per_seq": comparison_rows,
    }
    return row


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    flat_rows = []
    for row in rows:
        flat = {k: v for k, v in row.items() if k not in {"summary_thresholds", "per_seq"}}
        flat_rows.append(flat)
    fieldnames: list[str] = []
    for row in flat_rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(flat_rows)


def format_pct(value: Any) -> str:
    num = finite(value)
    if num is None:
        return "NA"
    return f"{num * 100:.2f}%"


def write_report(path: Path, rows: list[dict[str, Any]], final_taxonomy: str, decision_reason: str) -> None:
    lines = [
        "# ACL2 v114-TF HS-LQ Decision Summary",
        "",
        f"Final taxonomy: `{final_taxonomy}`",
        "",
        decision_reason,
        "",
        "Strict v114 pilot gate: median full ATE improvement >=5% or rolling p90 improvement >=5%, max full ATE harm <=1%, and segment scale not worse.",
        "",
        "| kind | label | median ATE | rolling p90 | max harm | segment scale | strict pass |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["kind"]),
                    str(row["label"]),
                    format_pct(row["median_full_ATE_rel_improvement"]),
                    format_pct(row["median_rolling_p90_rel_improvement"]),
                    format_pct(row["max_full_ATE_harm_rel"]),
                    format_pct(row["median_segment_scale_rel_improvement"]),
                    str(row["v114_strict_pilot_pass"]),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "Key evidence:",
            "",
            "- No semantic+internal candidate or lower-gain repair passed strict pilot.",
            "- `HS_LQ_CTRL_internal_confident_only_rerun1` was safe but weak: median ATE +1.34%, rolling +2.68%.",
            "- `HS_LQ_CTRL_rowmean_only_generic_scale_rerun1` had strong ATE/rolling gains but failed segment-scale safety: median segment-scale relative improvement -1.97%, 134/181 segment windows worse.",
            "- `HS_LQ_CTRL_rowmean_only_generic_scale_HS_MQ_risk_gate_mild_rerun1_full` passed 00/01/02/05 geometry and scale gates, but it is the generic rowmean/value-scaling control rather than a semantic+internal method.",
            "- Semantic+internal `HS_LQ3_stable_plus_risk_internal_quality_HS_MQ_risk_gate_mild_rerun1` still failed pilot.",
            "- Headwise semantic+internal MRT repairs, including the predicted-scale-delta-aware clip, improved scale safety but did not pass the strict ATE/rolling/harm gate.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    root = RESULTS_ROOT.resolve()
    diagnostics = root / "diagnostics"
    rows = [build_row(kind, label, prefix, diagnostics) for kind, label, prefix in RUNS]
    candidate_passes = [row for row in rows if row["kind"] in {"candidate", "repair"} and row["v114_strict_pilot_pass"]]
    rowmean = next(row for row in rows if row["label"] == "rowmean_only")
    rowmean_mrt_full = next((row for row in rows if row["label"] == "rowmean_MRT_risk_gate_full_00010205"), None)
    if candidate_passes:
        final_taxonomy = "HORIZONSTREAM_LOCAL_SEMANTIC_INTERNAL_SUCCESS"
        decision_reason = "At least one semantic+internal HS-LQ candidate passed strict pilot; downstream controls would be required before final method success."
    elif rowmean_mrt_full is not None and bool(rowmean_mrt_full["v114_strict_pilot_pass"]):
        final_taxonomy = "GEOMETRY_PASS_SEMANTIC_CAUSALITY_FAIL"
        decision_reason = "A generic rowmean/value scaling control plus MRT risk gate passed 00/01/02/05 geometry and scale gates, while semantic+internal candidates, including the predicted-scale-delta-aware MRT repair, failed pilot; this is a geometry baseline/confound, not semantic causality."
    elif (
        finite(rowmean["median_full_ATE_rel_improvement"]) is not None
        and finite(rowmean["median_rolling_p90_rel_improvement"]) is not None
        and float(rowmean["median_full_ATE_rel_improvement"]) >= ATE_THRESHOLD
        and float(rowmean["median_rolling_p90_rel_improvement"]) >= ROLLING_THRESHOLD
        and not bool(rowmean["segment_scale_not_worse_all"])
    ):
        final_taxonomy = "ALIGNMENT_SCALE_TRADEOFF_NO_GO"
        decision_reason = "The only strong geometry signal came from generic rowmean/value scaling, and it failed segment-scale safety; semantic+internal candidates were weak or sequence-harmful."
    else:
        final_taxonomy = "NO_GO_GOOD_HARM"
        decision_reason = "Semantic+internal candidates showed isolated positive sequence signal but did not satisfy strict pilot safety and median-gain thresholds."

    summary = {
        "schema": "acl2_v114tf_hs_lq_decision_summary_v1",
        "results_root": str(root),
        "thresholds": {
            "median_full_ATE_or_rolling_ge": ATE_THRESHOLD,
            "max_full_ATE_harm_rel_le": MAX_HARM_THRESHOLD,
            "segment_scale_not_worse_all": True,
        },
        "final_taxonomy": final_taxonomy,
        "decision_reason": decision_reason,
        "strict_pilot_pass_count": int(sum(bool(row["v114_strict_pilot_pass"]) for row in rows)),
        "rows": rows,
    }
    diagnostics.mkdir(parents=True, exist_ok=True)
    (diagnostics / "stage_hs_lq_decision_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_csv(diagnostics / "stage_hs_lq_decision_rows.csv", rows)
    write_report(root / "reports/HS_LQ_DECISION_SUMMARY.md", rows, final_taxonomy, decision_reason)
    print(json.dumps({"final_taxonomy": final_taxonomy, "strict_pilot_pass_count": summary["strict_pilot_pass_count"]}, sort_keys=True))


if __name__ == "__main__":
    main()
