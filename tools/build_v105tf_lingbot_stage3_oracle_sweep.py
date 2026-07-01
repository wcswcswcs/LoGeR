#!/usr/bin/env python3
"""Repair-space sweep for ACL2 v105-TF LingBot Stage 3 oracle diagnostics."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any, Callable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
STAGE3 = ROOT / "results/acl2_v105tf_dualtrack_lingbot_loger_evidence_eligibility/stage3_lingbot_oracle"
FRAME_ROWS = STAGE3 / "frame_semantic_geometry_rows.csv"


FEATURES = [
    "scale_reference_context_attention_frac",
    "local_window_context_attention_frac",
    "current_or_latest_frame_attention_frac",
    "semantic_scale_reference_attention_frac",
    "semantic_local_registration_attention_frac",
    "semantic_reject_unreliable_attention_frac",
    "scale_context_reject_attention_frac",
    "scale_context_structure_attention_frac",
    "local_context_reject_attention_frac",
    "scale_reference_patch_frac",
    "local_registration_patch_frac",
    "reject_unreliable_patch_frac",
    "semantic_confidence_p10",
]

SEMANTIC_FIELDS = {
    "semantic_scale_reference_attention_frac",
    "semantic_local_registration_attention_frac",
    "semantic_reject_unreliable_attention_frac",
    "scale_context_reject_attention_frac",
    "scale_context_structure_attention_frac",
    "local_context_reject_attention_frac",
    "scale_reference_patch_frac",
    "local_registration_patch_frac",
    "reject_unreliable_patch_frac",
    "semantic_confidence_p10",
}

CONTEXT_FIELDS = {
    "scale_reference_context_attention_frac",
    "local_window_context_attention_frac",
    "scale_context_reject_attention_frac",
    "scale_context_structure_attention_frac",
    "local_context_reject_attention_frac",
}


def parse_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def load_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    out: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        for feature in FEATURES + ["sim3_residual_m"]:
            item[feature] = float(item.get(feature, 0.0) or 0.0)
        item["bad_label"] = parse_bool(item.get("bad_label"))
        item["good_label"] = parse_bool(item.get("good_label"))
        out.append(item)
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def shifted_semantic_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = [dict(row) for row in rows]
    by_seq: dict[str, list[int]] = defaultdict(list)
    for idx, row in enumerate(rows):
        by_seq[str(row["seq"])].append(idx)
    for indices in by_seq.values():
        shifted = indices[1:] + indices[:1]
        for dst, src in zip(indices, shifted):
            for feature in SEMANTIC_FIELDS:
                out[dst][feature] = rows[src][feature]
    return out


def rotated_context_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = [dict(row) for row in rows]
    for row in out:
        row["scale_reference_context_attention_frac"], row["local_window_context_attention_frac"] = (
            row["local_window_context_attention_frac"],
            row["scale_reference_context_attention_frac"],
        )
        row["scale_context_reject_attention_frac"], row["local_context_reject_attention_frac"] = (
            row["local_context_reject_attention_frac"],
            row["scale_context_reject_attention_frac"],
        )
        row["scale_context_structure_attention_frac"] = 1.0 - float(row["scale_context_reject_attention_frac"])
    return out


def metric(rows: list[dict[str, Any]], selected: list[bool]) -> dict[str, Any]:
    bad = [i for i, row in enumerate(rows) if row["bad_label"]]
    good = [i for i, row in enumerate(rows) if row["good_label"]]
    selected_bad = [i for i in bad if selected[i]]
    selected_good = [i for i in good if selected[i]]
    bad_recall = len(selected_bad) / max(len(bad), 1)
    good_fpr = len(selected_good) / max(len(good), 1)
    return {
        "bad_recall": bad_recall,
        "good_FPR": good_fpr,
        "balanced_accuracy": 0.5 * (bad_recall + (1.0 - good_fpr)),
        "selected_rows": int(sum(1 for flag in selected if flag)),
        "selected_bad_rows": len(selected_bad),
        "selected_good_rows": len(selected_good),
        "selected_positive_sequence_coverage": len({str(rows[i]["seq"]) for i in selected_bad}),
    }


def random_metric(rows: list[dict[str, Any]], selected_count: int, trials: int = 256) -> dict[str, float]:
    rng = np.random.default_rng(1053)
    recalls = []
    fprs = []
    n = len(rows)
    for _ in range(trials):
        picks = set(int(x) for x in rng.choice(n, size=min(selected_count, n), replace=False))
        m = metric(rows, [idx in picks for idx in range(n)])
        recalls.append(float(m["bad_recall"]))
        fprs.append(float(m["good_FPR"]))
    return {
        "same_count_random_bad_recall_mean": float(np.mean(recalls)),
        "same_count_random_bad_recall_p95": float(np.percentile(recalls, 95)),
        "same_count_random_good_FPR_mean": float(np.mean(fprs)),
    }


def eval_policy(name: str, rows: list[dict[str, Any]], fn: Callable[[dict[str, Any]], bool], uses_semantic: bool, uses_context: bool) -> dict[str, Any]:
    selected = [fn(row) for row in rows]
    m = metric(rows, selected)
    rand = random_metric(rows, int(m["selected_rows"]))
    shifted = shifted_semantic_rows(rows) if uses_semantic else rows
    shifted_m = metric(shifted, [fn(row) for row in shifted])
    rotated = rotated_context_rows(rows) if uses_context else rows
    rotated_m = metric(rotated, [fn(row) for row in rotated])
    out = {
        "schema": "acl2_v105tf_lingbot_stage3_oracle_sweep_metric_v1",
        "policy": name,
        **m,
        **rand,
        "same_count_random_margin": float(m["bad_recall"]) - float(rand["same_count_random_bad_recall_mean"]),
        "semantic_shuffle_bad_recall": shifted_m["bad_recall"],
        "semantic_shuffle_margin": float(m["bad_recall"]) - float(shifted_m["bad_recall"]),
        "context_role_rotation_bad_recall": rotated_m["bad_recall"],
        "context_role_rotation_margin": float(m["bad_recall"]) - float(rotated_m["bad_recall"]),
        "uses_semantic": uses_semantic,
        "uses_context": uses_context,
    }
    out["stage3_oracle_pass"] = (
        float(out["bad_recall"]) >= 0.65
        and float(out["good_FPR"]) <= 0.25
        and int(out["selected_positive_sequence_coverage"]) >= 2
        and float(out["same_count_random_margin"]) >= 0.05
        and (not uses_semantic or float(out["semantic_shuffle_margin"]) >= 0.05)
    )
    return out


def feature_thresholds(rows: list[dict[str, Any]], feature: str) -> list[float]:
    vals = np.asarray([float(row[feature]) for row in rows], dtype=np.float64)
    qs = [5, 10, 15, 20, 25, 33, 40, 50, 60, 67, 75, 80, 85, 90, 95]
    return sorted({float(np.percentile(vals, q)) for q in qs})


def build() -> dict[str, Any]:
    rows = load_rows(FRAME_ROWS)
    metrics: list[dict[str, Any]] = []

    for feature in FEATURES:
        uses_semantic = feature in SEMANTIC_FIELDS
        uses_context = feature in CONTEXT_FIELDS
        for thr in feature_thresholds(rows, feature):
            metrics.append(
                eval_policy(
                    f"{feature}_ge_{thr:.6g}",
                    rows,
                    lambda row, feature=feature, thr=thr: float(row[feature]) >= thr,
                    uses_semantic,
                    uses_context,
                )
            )
            metrics.append(
                eval_policy(
                    f"{feature}_le_{thr:.6g}",
                    rows,
                    lambda row, feature=feature, thr=thr: float(row[feature]) <= thr,
                    uses_semantic,
                    uses_context,
                )
            )

    combo_features = [
        "reject_unreliable_patch_frac",
        "scale_reference_patch_frac",
        "semantic_reject_unreliable_attention_frac",
        "semantic_scale_reference_attention_frac",
        "scale_context_reject_attention_frac",
        "scale_context_structure_attention_frac",
        "scale_reference_context_attention_frac",
        "local_window_context_attention_frac",
    ]
    for f1, f2 in combinations(combo_features, 2):
        for t1 in feature_thresholds(rows, f1)[::2]:
            for t2 in feature_thresholds(rows, f2)[::2]:
                uses_semantic = f1 in SEMANTIC_FIELDS or f2 in SEMANTIC_FIELDS
                uses_context = f1 in CONTEXT_FIELDS or f2 in CONTEXT_FIELDS
                metrics.append(
                    eval_policy(
                        f"{f1}_ge_{t1:.4g}_AND_{f2}_ge_{t2:.4g}",
                        rows,
                        lambda row, f1=f1, f2=f2, t1=t1, t2=t2: float(row[f1]) >= t1 and float(row[f2]) >= t2,
                        uses_semantic,
                        uses_context,
                    )
                )
                metrics.append(
                    eval_policy(
                        f"{f1}_le_{t1:.4g}_AND_{f2}_le_{t2:.4g}",
                        rows,
                        lambda row, f1=f1, f2=f2, t1=t1, t2=t2: float(row[f1]) <= t1 and float(row[f2]) <= t2,
                        uses_semantic,
                        uses_context,
                    )
                )

    metrics.sort(
        key=lambda row: (
            row["stage3_oracle_pass"],
            float(row["balanced_accuracy"]),
            float(row["bad_recall"]),
            -float(row["good_FPR"]),
        ),
        reverse=True,
    )
    for row in metrics:
        row["stage3_oracle_pass"] = "true" if row["stage3_oracle_pass"] else "false"

    pass_rows = [row for row in metrics if row["stage3_oracle_pass"] == "true"]
    summary = {
        "schema": "acl2_v105tf_lingbot_stage3_oracle_sweep_summary_v1",
        "candidate_policy_count": len(metrics),
        "passing_policy_count": len(pass_rows),
        "best_policy": metrics[0]["policy"] if metrics else "",
        "best_balanced_accuracy": metrics[0]["balanced_accuracy"] if metrics else 0.0,
        "best_bad_recall": metrics[0]["bad_recall"] if metrics else 0.0,
        "best_good_FPR": metrics[0]["good_FPR"] if metrics else 0.0,
        "stage3_sweep_pass": bool(pass_rows),
        "passing_policies": [row["policy"] for row in pass_rows[:20]],
        "note": "Diagnostic repair-space sweep over trace32 rows; no runtime action was run.",
    }
    write_csv(STAGE3 / "oracle_policy_sweep_metrics.csv", metrics)
    write_text(STAGE3 / "oracle_policy_sweep_summary.json", json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return summary


if __name__ == "__main__":
    build()
