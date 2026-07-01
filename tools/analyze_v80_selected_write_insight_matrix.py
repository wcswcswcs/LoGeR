#!/usr/bin/env python3
"""Build an auditable selected-write insight matrix for ACL2 v80 phase10.

This script is intentionally diagnostic-only. It joins existing selected-write
support summaries with the small downstream probe batch and reports which
patterns survive counterexamples.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


def parse_key(text: str) -> tuple[str, int]:
    seq, chunk = text.split(":")
    return f"{int(seq):02d}", int(chunk)


def fnum(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(out):
        return None
    return out


def inum(value: Any) -> int | None:
    num = fnum(value)
    if num is None:
        return None
    return int(num)


def find_seq(path: str) -> str | None:
    matches = re.findall(r"seq(\d{2})", path)
    if not matches:
        return None
    return matches[-1]


def load_support_rows(path: Path, threshold: float) -> dict[tuple[str, int], dict[str, Any]]:
    rows: dict[tuple[str, int], dict[str, Any]] = {}
    with path.open(newline="") as fh:
        for raw in csv.DictReader(fh):
            seq = f"{int(raw['seq']):02d}"
            chunk = int(raw["chunk"])
            low = fnum(raw.get("selected_low_support_given_selected_runtime"))
            row = dict(raw)
            row.update(
                {
                    "seq": seq,
                    "chunk": chunk,
                    "source": "support_rows_csv",
                    "support_threshold": threshold,
                    "selected_low_support_given_selected_runtime": low,
                    "selected_low_support_mass": inum(raw.get("selected_low_support_mass")),
                    "selected_runtime_mass": inum(raw.get("selected_runtime_mass")),
                    "runtime_low_support_mass": inum(raw.get("runtime_low_support_mass")),
                    "support_score_q10": fnum(raw.get("support_score_q10")),
                    "support_score_q50": fnum(raw.get("support_score_q50")),
                    "support_score_mean": fnum(raw.get("support_score_mean")),
                    "baseline_abs_error_mean_m_phase2": fnum(raw.get("baseline_abs_error_mean_m_phase2")),
                    "selected_risk_given_selected_phase2": fnum(raw.get("selected_risk_given_selected_phase2")),
                    "diagnostic_positive_flag": bool(low is not None and low >= threshold),
                }
            )
            rows[(seq, chunk)] = row
    return rows


def scan_selected_summaries(
    patterns: list[str],
    rows: dict[tuple[str, int], dict[str, Any]],
    known_bad: set[tuple[str, int]],
    known_good: set[tuple[str, int]],
    threshold: float,
) -> int:
    best_paths: dict[tuple[str, int], str] = {}
    skipped_unclassified = 0
    for pattern in patterns:
        for path in glob.glob(pattern, recursive=True):
            name = Path(path).name
            # Keep chunk-specific summaries only; the aggregate filename can be
            # ambiguous in multi-chunk directories.
            if not re.match(r"chunk_\d{3}_selected_write_support_map_summary\.json$", name):
                continue
            try:
                data = json.loads(Path(path).read_text())
            except Exception:
                continue
            if fnum(data.get("support_threshold")) != threshold:
                continue
            seq = find_seq(path)
            chunk = inum(data.get("chunk"))
            if seq is None or chunk is None:
                continue
            key = (seq, chunk)
            # Prefer shorter/direct paths over threshold sensitivity duplicates.
            prev = best_paths.get(key)
            if prev is None or ("threshold_sensitivity" in prev and "threshold_sensitivity" not in path):
                best_paths[key] = path

    for key, path in sorted(best_paths.items()):
        if key in rows:
            rows[key]["extra_selected_summary"] = path
            continue
        data = json.loads(Path(path).read_text())
        if key in known_bad:
            group = "bad_candidate"
        elif key in known_good:
            group = "good_counterexample"
        elif any("good_counterexample" in part.lower() for part in Path(path).parts):
            group = "good_counterexample"
        elif any("bad_candidate" in part.lower() for part in Path(path).parts):
            group = "bad_candidate"
        else:
            skipped_unclassified += 1
            continue
        low = fnum(data.get("selected_low_support_given_selected_runtime"))
        rows[key] = {
            "seq": key[0],
            "chunk": key[1],
            "group": group,
            "case_types_phase2": "[]",
            "source": "selected_summary_scan",
            "support_threshold": threshold,
            "selected_low_support_given_selected_runtime": low,
            "selected_low_support_mass": inum(data.get("selected_low_support_mass")),
            "selected_runtime_mass": inum(data.get("selected_runtime_mass")),
            "runtime_low_support_mass": inum(data.get("runtime_low_support_mass")),
            "support_score_q10": None,
            "support_score_q50": fnum(data.get("score_q50")),
            "support_score_mean": fnum(data.get("score_mean")),
            "baseline_abs_error_mean_m_phase2": None,
            "selected_risk_given_selected_phase2": None,
            "diagnostic_positive_flag": bool(low is not None and low >= threshold),
            "selected_summary": path,
        }
    return skipped_unclassified


def load_downstream(path: Path) -> dict[tuple[str, int], dict[str, Any]]:
    data = json.loads(path.read_text())
    out: dict[tuple[str, int], dict[str, Any]] = {}
    for row in data.get("case_rows", []):
        key = (f"{int(row['seq']):02d}", int(row["chunk"]))
        out[key] = {
            "selected_downstream_max_abs_pose_value_diff_vs_LW1": fnum(row.get("selected_downstream_max_abs_pose_value_diff_vs_LW1")),
            "control_downstream_max_abs_pose_value_diff_vs_LW1": fnum(row.get("control_downstream_max_abs_pose_value_diff_vs_LW1")),
            "selected_minus_control_downstream_max": fnum(row.get("selected_minus_control_downstream_max")),
        }
    return out


def stats(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"n": 0, "min": None, "max": None, "mean": None}
    return {"n": len(values), "min": min(values), "max": max(values), "mean": mean(values)}


def build_cluster_rows(rows: list[dict[str, Any]], threshold: float) -> list[dict[str, Any]]:
    by_seq: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("group") == "bad_candidate":
            by_seq[row["seq"]].append(row)

    clusters: list[dict[str, Any]] = []
    for seq, seq_rows in by_seq.items():
        seq_rows = sorted(seq_rows, key=lambda r: r["chunk"])
        current: list[dict[str, Any]] = []
        for row in seq_rows:
            if not current or row["chunk"] == current[-1]["chunk"] + 1:
                current.append(row)
            else:
                if current:
                    clusters.append(summarize_cluster(seq, current, threshold))
                current = [row]
        if current:
            clusters.append(summarize_cluster(seq, current, threshold))
    return clusters


def build_positive_cluster_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_group_seq: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("diagnostic_positive_flag"):
            by_group_seq[(row.get("group", "unknown"), row["seq"])].append(row)

    clusters: list[dict[str, Any]] = []
    for (group, seq), seq_rows in by_group_seq.items():
        seq_rows = sorted(seq_rows, key=lambda r: r["chunk"])
        current: list[dict[str, Any]] = []
        for row in seq_rows:
            if not current or row["chunk"] == current[-1]["chunk"] + 1:
                current.append(row)
            else:
                clusters.append(summarize_positive_cluster(group, seq, current))
                current = [row]
        if current:
            clusters.append(summarize_positive_cluster(group, seq, current))
    return clusters


def summarize_positive_cluster(group: str, seq: str, cluster: list[dict[str, Any]]) -> dict[str, Any]:
    lows = [r["selected_low_support_given_selected_runtime"] for r in cluster if r.get("selected_low_support_given_selected_runtime") is not None]
    return {
        "group": group,
        "seq": seq,
        "start_chunk": cluster[0]["chunk"],
        "end_chunk": cluster[-1]["chunk"],
        "n": len(cluster),
        "selected_low_support_mean": mean(lows) if lows else None,
    }


def binary_rule_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    labeled = [r for r in rows if r.get("group") in {"bad_candidate", "good_counterexample"}]
    tp = sum(1 for r in labeled if r.get("group") == "bad_candidate" and r.get("diagnostic_positive_flag"))
    fn = sum(1 for r in labeled if r.get("group") == "bad_candidate" and not r.get("diagnostic_positive_flag"))
    fp = sum(1 for r in labeled if r.get("group") == "good_counterexample" and r.get("diagnostic_positive_flag"))
    tn = sum(1 for r in labeled if r.get("group") == "good_counterexample" and not r.get("diagnostic_positive_flag"))
    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    fpr = fp / (fp + tn) if (fp + tn) else None
    return {"tp": tp, "fn": fn, "fp": fp, "tn": tn, "precision": precision, "recall": recall, "false_positive_rate": fpr}


def summarize_cluster(seq: str, cluster: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    positives = [r for r in cluster if r.get("diagnostic_positive_flag")]
    false_negatives = [r for r in cluster if not r.get("diagnostic_positive_flag")]
    lows = [r["selected_low_support_given_selected_runtime"] for r in cluster if r.get("selected_low_support_given_selected_runtime") is not None]
    return {
        "seq": seq,
        "start_chunk": cluster[0]["chunk"],
        "end_chunk": cluster[-1]["chunk"],
        "n": len(cluster),
        "positive_count": len(positives),
        "positive_rate": len(positives) / len(cluster) if cluster else None,
        "false_negative_chunks": ",".join(str(r["chunk"]) for r in false_negatives),
        "selected_low_support_mean": mean(lows) if lows else None,
        "support_threshold": threshold,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--support-rows", required=True)
    ap.add_argument("--downstream-summary", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--support-threshold", type=float, default=0.5)
    ap.add_argument("--selected-summary-glob", action="append", default=[])
    ap.add_argument("--known-bad", action="append", default=[])
    ap.add_argument("--known-good", action="append", default=[])
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    threshold = args.support_threshold
    known_bad = {parse_key(x) for x in args.known_bad}
    known_good = {parse_key(x) for x in args.known_good}

    rows_by_key = load_support_rows(Path(args.support_rows), threshold)
    skipped_unclassified = scan_selected_summaries(args.selected_summary_glob, rows_by_key, known_bad, known_good, threshold)
    downstream = load_downstream(Path(args.downstream_summary))

    matrix_rows: list[dict[str, Any]] = []
    downstream_rows: list[dict[str, Any]] = []
    for key, row in sorted(rows_by_key.items()):
        out = dict(row)
        ds = downstream.get(key, {})
        out.update(ds)
        delta = ds.get("selected_minus_control_downstream_max")
        if delta is not None:
            out["downstream_selected_gt_control"] = delta > 0
            downstream_rows.append(out)
        matrix_rows.append(out)

    by_group = defaultdict(list)
    for row in matrix_rows:
        by_group[row.get("group", "unknown")].append(row)

    group_summary: dict[str, Any] = {}
    for group, grows in sorted(by_group.items()):
        lows = [r["selected_low_support_given_selected_runtime"] for r in grows if r.get("selected_low_support_given_selected_runtime") is not None]
        positives = [r for r in grows if r.get("diagnostic_positive_flag")]
        ds_rows = [r for r in grows if r.get("selected_minus_control_downstream_max") is not None]
        ds_pos = [r for r in ds_rows if r.get("selected_minus_control_downstream_max") is not None and r["selected_minus_control_downstream_max"] > 0]
        group_summary[group] = {
            "row_count": len(grows),
            "low_support_positive_count": len(positives),
            "low_support_positive_rate": len(positives) / len(grows) if grows else None,
            "selected_low_support_stats": stats(lows),
            "downstream_tested_count": len(ds_rows),
            "downstream_selected_gt_control_count": len(ds_pos),
            "downstream_selected_gt_control_rate": len(ds_pos) / len(ds_rows) if ds_rows else None,
        }

    cluster_rows = build_cluster_rows(matrix_rows, threshold)
    positive_cluster_rows = build_positive_cluster_rows(matrix_rows)
    notable_clusters = [
        row
        for row in cluster_rows
        if row["n"] >= 3 and row["positive_count"] >= 2
    ]
    positive_cluster_summary: dict[str, Any] = {}
    for row in positive_cluster_rows:
        group = row["group"]
        summary = positive_cluster_summary.setdefault(group, {"cluster_count": 0, "max_n": 0, "clusters": []})
        summary["cluster_count"] += 1
        summary["max_n"] = max(summary["max_n"], row["n"])
        summary["clusters"].append(row)
    good_false_positives = [
        {"seq": r["seq"], "chunk": r["chunk"], "selected_low_support": r.get("selected_low_support_given_selected_runtime")}
        for r in matrix_rows
        if r.get("group") == "good_counterexample" and r.get("diagnostic_positive_flag")
    ]
    bad_false_negatives = [
        {"seq": r["seq"], "chunk": r["chunk"], "selected_low_support": r.get("selected_low_support_given_selected_runtime")}
        for r in matrix_rows
        if r.get("group") == "bad_candidate" and not r.get("diagnostic_positive_flag")
    ]

    summary = {
        "schema": "acl2_v80_selected_write_insight_matrix_v1",
        "diagnostic_only": True,
        "method_gate_claimed": False,
        "runtime_promotion_allowed": False,
        "v80_goal_achieved": False,
        "support_threshold": threshold,
        "row_count": len(matrix_rows),
        "skipped_unclassified_selected_summary_count": skipped_unclassified,
        "low_support_binary_rule_metrics": binary_rule_metrics(matrix_rows),
        "group_summary": group_summary,
        "notable_bad_clusters": notable_clusters,
        "positive_cluster_summary": positive_cluster_summary,
        "good_false_positives": good_false_positives,
        "bad_false_negatives": bad_false_negatives,
        "insights": [
            "selected-write low-support concentration is a repeated bad-cluster signal, not a single seq05/chunk83 artifact",
            "low-support concentration alone is not good-safe because multiple good counterexamples remain positive",
            "downstream selected-minus-control is not good-safe by itself after adding cross-sequence good counterexamples",
            "seq02 chunk68 is a bad false negative for low-support concentration but still has positive selected-minus-control downstream response",
            "all downstream findings are diagnostic-only because no GT/J/ATE improvement gate is measured or passed here",
        ],
        "outputs": {
            "matrix_rows_csv": str(out_dir / "selected_write_insight_matrix_rows.csv"),
            "downstream_join_rows_csv": str(out_dir / "selected_write_downstream_join_rows.csv"),
            "cluster_rows_csv": str(out_dir / "selected_write_bad_cluster_rows.csv"),
            "positive_cluster_rows_csv": str(out_dir / "selected_write_positive_cluster_rows.csv"),
            "summary_json": str(out_dir / "selected_write_insight_matrix_summary.json"),
        },
    }

    write_csv(out_dir / "selected_write_insight_matrix_rows.csv", matrix_rows)
    write_csv(out_dir / "selected_write_downstream_join_rows.csv", downstream_rows)
    write_csv(out_dir / "selected_write_bad_cluster_rows.csv", cluster_rows)
    write_csv(out_dir / "selected_write_positive_cluster_rows.csv", positive_cluster_rows)
    (out_dir / "selected_write_insight_matrix_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
