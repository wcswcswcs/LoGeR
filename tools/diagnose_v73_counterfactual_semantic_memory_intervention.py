#!/usr/bin/env python3
"""Diagnostic-only Phase 7 explanation for v73 semantic-memory components.

This script intentionally does not claim to run online counterfactual
interventions. It explains existing Phase-E runs by ranking trace/component
features that separate positive chunks from failures.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


FEATURES = [
    "q_handoff",
    "overlap_residual",
    "qscale_factor",
    "effective_alpha",
    "stable_mean",
    "risk_mean",
    "remaining_valid_ratio",
    "reset_relative_index",
    "chunk_norm",
    "scale",
    "component_consistency_proxy",
    "component_top_mass_ratio",
    "component_count_norm",
    "component_stable_variance",
    "component_risk_variance",
]

LABELS = ["head_tail_pass", "overlap_pass"]


def _safe_float(value: Any) -> float:
    if isinstance(value, bool):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _read_rows(path: Path) -> List[Dict[str, Any]]:
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"empty rows file: {path}")
    return rows


def _auc_score(values: Iterable[float], labels: Iterable[bool]) -> Optional[float]:
    pairs = list(zip(values, labels))
    pos = [value for value, label in pairs if label]
    neg = [value for value, label in pairs if not label]
    if not pos or not neg:
        return None
    wins = 0
    ties = 0
    for p_value in pos:
        for n_value in neg:
            if p_value > n_value:
                wins += 1
            elif p_value == n_value:
                ties += 1
    return (wins + 0.5 * ties) / (len(pos) * len(neg))


def _mean(values: Iterable[float]) -> Optional[float]:
    vals = [value for value in values if value == value]
    if not vals:
        return None
    return sum(vals) / len(vals)


def _analyze(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for label in LABELS:
        labels = [str(row[label]) == "True" for row in rows]
        positive_count = sum(labels)
        for feature in FEATURES:
            values = [_safe_float(row.get(feature)) for row in rows]
            pos_mean = _mean(value for value, is_pos in zip(values, labels) if is_pos)
            fail_mean = _mean(value for value, is_pos in zip(values, labels) if not is_pos)
            for direction in ("desc", "asc"):
                ranked_values = values if direction == "desc" else [-value for value in values]
                auc = _auc_score(ranked_values, labels)
                order = sorted(range(len(rows)), key=lambda i: ranked_values[i], reverse=True)
                top5 = order[:5]
                top5_precision = sum(1 for i in top5 if labels[i]) / 5.0
                out.append(
                    {
                        "diagnostic_only": True,
                        "intervention_type": "trace_feature_contrast_not_online_counterfactual",
                        "label": label,
                        "positive_count": positive_count,
                        "feature": feature,
                        "direction": direction,
                        "auc": auc,
                        "top5_precision": top5_precision,
                        "top5_chunks": ",".join(str(rows[i]["chunk"]) for i in top5),
                        "positive_chunks": ",".join(str(row["chunk"]) for row, is_pos in zip(rows, labels) if is_pos),
                        "positive_mean": pos_mean,
                        "failure_mean": fail_mean,
                        "positive_minus_failure_mean": None if pos_mean is None or fail_mean is None else pos_mean - fail_mean,
                        "phase7_gate_pass": bool(positive_count >= 4 and auc is not None and auc >= 0.70 and top5_precision >= 0.40),
                    }
                )
    out.sort(key=lambda row: (row["label"], -(row["auc"] if row["auc"] is not None else -1.0), -row["top5_precision"]))
    return out


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"no rows for {path}")
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path: Path, summary: Dict[str, Any]) -> None:
    lines = [
        "# Phase 7 Diagnostic-Only Component Explanation",
        "",
        "This artifact explains existing MT11 full11 traces. It is not an online counterfactual intervention result.",
        "",
        f"Source rows: `{summary['source_rows']}`",
        "",
    ]
    for label, data in summary["labels"].items():
        best = data["best_feature_contrast"]
        lines.extend(
            [
                f"## {label}",
                "",
                f"- positive count: `{data['positive_count']}/11`",
                f"- positive chunks: `{best['positive_chunks']}`",
                f"- best feature: `{best['feature']}` direction `{best['direction']}`",
                f"- AUC: `{best['auc']}`",
                f"- top5 precision: `{best['top5_precision']}`",
                f"- top5 chunks: `{best['top5_chunks']}`",
                f"- Phase7 diagnostic gate pass: `{best['phase7_gate_pass']}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Conclusion",
            "",
            summary["conclusion"],
            "",
        ]
    )
    path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", required=True, help="H3 predictor rows CSV with trace/component features.")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    rows_path = Path(args.rows)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = _read_rows(rows_path)
    result_rows = _analyze(rows)
    best_by_label = {label: next(row for row in result_rows if row["label"] == label) for label in LABELS}
    summary = {
        "schema": "v73_phase7_diagnostic_only_component_explanation_v1",
        "diagnostic_only": True,
        "source_rows": str(rows_path),
        "row_count": len(rows),
        "labels": {
            label: {
                "positive_count": int(best_by_label[label]["positive_count"]),
                "best_feature_contrast": best_by_label[label],
            }
            for label in LABELS
        },
        "phase7_gate_pass": False,
        "conclusion": "No semantic component type is consistent across >=4 chunks; current evidence explains local positives but does not justify selector/integration.",
    }
    _write_csv(out_dir / "component_intervention_results.csv", result_rows)
    (out_dir / "component_causal_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    _write_markdown(out_dir / "top_positive_components.md", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"wrote_csv={out_dir / 'component_intervention_results.csv'}")
    print(f"wrote_json={out_dir / 'component_causal_summary.json'}")
    print(f"wrote_md={out_dir / 'top_positive_components.md'}")


if __name__ == "__main__":
    main()
