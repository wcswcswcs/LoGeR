#!/usr/bin/env python3
"""Aggregate ACL2 v35B paired no-commit probe rows."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from statistics import median
from typing import Any, Dict, Iterable, List, Optional, Tuple


RUN_RE = re.compile(r"V35B_TRACKA_(?:C)?PROBE_R\d+_(?P<parent>H9|C9)_chunk(?P<chunk>\d+)_(?P<cue>orig|sem)$")
ORACLE_POSITIVE_CHUNKS = {0, 10}


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def mean(values: Iterable[Optional[float]]) -> Optional[float]:
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not vals:
        return None
    return sum(vals) / len(vals)


def rms(values: Iterable[float]) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    if not vals:
        return 0.0
    return math.sqrt(sum(v * v for v in vals) / len(vals))


def mad(values: List[float]) -> float:
    if not values:
        return 0.0
    m = median(values)
    return median([abs(v - m) for v in values])


def run_done(run_dir: Path, run_name: str) -> bool:
    status = run_dir / "run_status.txt"
    return status.exists() and f"DONE {run_name}" in status.read_text(encoding="utf-8", errors="replace")


def summarize_run(run_dir: Path, run_name: str, parent: str, chunk: int, cue: str) -> Dict[str, Any]:
    rows = read_jsonl(run_dir / "hmc_state_hash.jsonl")
    return {
        "run_name": run_name,
        "parent": parent,
        "chunk": int(chunk),
        "cue": cue,
        "done": run_done(run_dir, run_name),
        "rows": len(rows),
        "all_probe_no_commit": bool(rows) and all(bool(r.get("probe_no_commit_hash_equal")) for r in rows),
        "first_hash_before": rows[0].get("hash_H_m_before_probe") if rows else "",
        "first_hash_after": rows[0].get("hash_H_m_after_probe") if rows else "",
        "mean_D_patch": mean(r.get("prior_mean_D_patch") for r in rows),
        "q90_D_patch": mean(r.get("prior_q90_D_patch") for r in rows),
        "mass_D_gt_050": mean(r.get("prior_dynamic_mass_D_gt_050") for r in rows),
        "label_count_mean": mean(r.get("prior_v31_semantic_label_count") for r in rows),
        "fallback_mean": mean(r.get("prior_v31_semantic_label_fallback_ratio") for r in rows),
        "semantic_d_mean": mean(r.get("prior_v32_semantic_d_mean") for r in rows),
        "semantic_d_q90": mean(r.get("prior_v32_semantic_d_q90") for r in rows),
        "semantic_applied_rows": sum(1 for r in rows if bool(r.get("prior_v31_semantic_recondition_applied"))),
        "pose_t_mean": mean(r.get("pass1_pass2_pose_t_mean") for r in rows),
        "pose_t_max": mean(r.get("pass1_pass2_pose_t_max") for r in rows),
        "pose_r_deg_mean": mean(r.get("pass1_pass2_pose_r_deg_mean") for r in rows),
        "hash_H_next_equals_before": bool(rows) and all(
            r.get("hash_H_next") == r.get("hash_H_m_before_probe") for r in rows
        ),
    }


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n", extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollout-root", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    rollout_root = Path(args.rollout_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    run_rows: List[Dict[str, Any]] = []
    by_key: Dict[Tuple[str, int, str], Dict[str, Any]] = {}
    for d in sorted(rollout_root.iterdir() if rollout_root.exists() else []):
        if not d.is_dir():
            continue
        m = RUN_RE.match(d.name)
        if not m:
            continue
        parent = m.group("parent")
        chunk = int(m.group("chunk"))
        cue = m.group("cue")
        row = summarize_run(d, d.name, parent, chunk, cue)
        run_rows.append(row)
        by_key[(parent, chunk, cue)] = row

    pair_rows: List[Dict[str, Any]] = []
    for parent in ["H9", "C9"]:
        for chunk in [0, 5, 10, 15, 20, 25, 30]:
            a = by_key.get((parent, chunk, "orig"))
            b = by_key.get((parent, chunk, "sem"))
            if not a or not b:
                pair_rows.append({
                    "parent": parent,
                    "chunk": chunk,
                    "missing_pair": True,
                })
                continue
            deltas = {
                "delta_mean_D": (b["mean_D_patch"] or 0.0) - (a["mean_D_patch"] or 0.0),
                "delta_q90_D": (b["q90_D_patch"] or 0.0) - (a["q90_D_patch"] or 0.0),
                "delta_mass_D_gt_050": (b["mass_D_gt_050"] or 0.0) - (a["mass_D_gt_050"] or 0.0),
                "delta_semantic_d_mean": (b["semantic_d_mean"] or 0.0) - (a["mean_D_patch"] or 0.0),
            }
            pair_rows.append({
                "parent": parent,
                "chunk": chunk,
                "missing_pair": False,
                "orig_done": a["done"],
                "sem_done": b["done"],
                "state_hash_before_equal": a["first_hash_before"] == b["first_hash_before"],
                "orig_no_commit": a["all_probe_no_commit"],
                "sem_no_commit": b["all_probe_no_commit"],
                "probe_no_commit_exact": bool(a["all_probe_no_commit"] and b["all_probe_no_commit"]),
                "orig_rows": a["rows"],
                "sem_rows": b["rows"],
                "orig_mean_D": a["mean_D_patch"],
                "sem_mean_D": b["mean_D_patch"],
                "orig_q90_D": a["q90_D_patch"],
                "sem_q90_D": b["q90_D_patch"],
                "orig_mass_D_gt_050": a["mass_D_gt_050"],
                "sem_mass_D_gt_050": b["mass_D_gt_050"],
                "sem_label_count_mean": b["label_count_mean"],
                "sem_fallback_mean": b["fallback_mean"],
                "sem_applied_rows": b["semantic_applied_rows"],
                "orig_pose_t_mean": a["pose_t_mean"],
                "sem_pose_t_mean": b["pose_t_mean"],
                "orig_pose_t_max": a["pose_t_max"],
                "sem_pose_t_max": b["pose_t_max"],
                "orig_pose_r_deg_mean": a["pose_r_deg_mean"],
                "sem_pose_r_deg_mean": b["pose_r_deg_mean"],
                "hash_H_next_equals_before": a["hash_H_next_equals_before"] and b["hash_H_next_equals_before"],
                "D_delta_rms_proxy": rms(deltas.values()),
                **deltas,
                "delta_pose_t_mean": (b["pose_t_mean"] or 0.0) - (a["pose_t_mean"] or 0.0),
                "delta_pose_t_max": (b["pose_t_max"] or 0.0) - (a["pose_t_max"] or 0.0),
                "delta_pose_r_deg_mean": (b["pose_r_deg_mean"] or 0.0) - (a["pose_r_deg_mean"] or 0.0),
                "oracle_positive_diagnostic": chunk in ORACLE_POSITIVE_CHUNKS,
            })

    valid_pairs = [r for r in pair_rows if not r.get("missing_pair")]
    delta_vals = [float(r["D_delta_rms_proxy"]) for r in valid_pairs]
    threshold = median(delta_vals) + 0.5 * mad(delta_vals) if delta_vals else float("inf")
    q90_threshold = median([float(r["delta_q90_D"]) for r in valid_pairs]) if valid_pairs else float("inf")
    mass_threshold = median([float(r["delta_mass_D_gt_050"]) for r in valid_pairs]) if valid_pairs else float("inf")

    decisions: List[Dict[str, Any]] = []
    for r in valid_pairs:
        label_ok = (r.get("sem_label_count_mean") is not None and float(r["sem_label_count_mean"]) >= 1.0
                    and (r.get("sem_fallback_mean") is not None and float(r["sem_fallback_mean"]) <= 0.10))
        has_controlled_pose = r.get("orig_pose_t_max") is not None and r.get("sem_pose_t_max") is not None
        if has_controlled_pose:
            improve_count = int(float(r["delta_pose_t_max"]) <= -0.01)
            improve_count += int(float(r["D_delta_rms_proxy"]) >= 0.15)
            improve_count += int(float(r["delta_mass_D_gt_050"]) >= 0.10)
            hard_regress_count = int(float(r["delta_pose_t_mean"]) > 0.02)
            hard_regress_count += int(float(r["delta_pose_r_deg_mean"]) > 1.0)
            hard_regress_count += int(not bool(r.get("hash_H_next_equals_before")))
        else:
            improve_count = int(float(r["D_delta_rms_proxy"]) >= threshold)
            improve_count += int(float(r["delta_q90_D"]) >= q90_threshold)
            improve_count += int(float(r["delta_mass_D_gt_050"]) >= mass_threshold)
            hard_regress_count = 0
        hard_regress_count += int(not r["probe_no_commit_exact"]) + int(not r["state_hash_before_equal"]) + int(not label_ok)
        g_probe = bool(improve_count >= 3 and hard_regress_count == 0)
        decisions.append({
            **r,
            "robust_delta_threshold": threshold,
            "robust_q90_delta_threshold": q90_threshold,
            "robust_mass_delta_threshold": mass_threshold,
            "label_ok": label_ok,
            "improve_count": improve_count,
            "hard_regress_count": hard_regress_count,
            "G_probe": g_probe,
        })

    positives = [d for d in decisions if d["oracle_positive_diagnostic"]]
    negatives = [d for d in decisions if not d["oracle_positive_diagnostic"]]
    tp = sum(1 for d in positives if d["G_probe"])
    fn = sum(1 for d in positives if not d["G_probe"])
    fp = sum(1 for d in negatives if d["G_probe"])
    tn = sum(1 for d in negatives if not d["G_probe"])
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    summary = {
        "paired_probe_rows": len(run_rows),
        "pairs": len(valid_pairs),
        "missing_pairs": sum(1 for r in pair_rows if r.get("missing_pair")),
        "all_runs_done": bool(run_rows) and all(bool(r["done"]) for r in run_rows),
        "all_probe_no_commit_exact": bool(valid_pairs) and all(bool(r["probe_no_commit_exact"]) for r in valid_pairs),
        "all_hash_H_next_equals_before": bool(valid_pairs) and all(
            bool(r.get("hash_H_next_equals_before", True)) for r in valid_pairs
        ),
        "all_pair_state_hash_before_equal": bool(valid_pairs) and all(bool(r["state_hash_before_equal"]) for r in valid_pairs),
        "uses_absolute_chunk_id": False,
        "oracle_positive_recall": recall,
        "false_positive_rate": fpr,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "G_probe_positive_chunks": sorted({int(d["chunk"]) for d in decisions if d["G_probe"]}),
        "gate_pass": bool(recall >= 0.75 and fpr <= 0.25 and all(bool(r["probe_no_commit_exact"]) for r in valid_pairs)),
        "rule": {
            "type": "training_free_robust_stat_or_controlled_pose_guard",
            "improve_count_required": 3,
            "hard_regress_count_required": 0,
            "D_delta_rms_threshold": threshold,
            "q90_delta_threshold": q90_threshold,
            "mass_delta_threshold": mass_threshold,
        },
    }

    write_csv(out_dir / "paired_probe_run_metrics.csv", run_rows, [
        "run_name", "parent", "chunk", "cue", "done", "rows", "all_probe_no_commit",
        "first_hash_before", "first_hash_after", "mean_D_patch", "q90_D_patch",
        "mass_D_gt_050", "label_count_mean", "fallback_mean", "semantic_d_mean",
        "semantic_d_q90", "semantic_applied_rows", "pose_t_mean", "pose_t_max",
        "pose_r_deg_mean", "hash_H_next_equals_before",
    ])
    write_csv(out_dir / "paired_probe_metrics.csv", pair_rows, [
        "parent", "chunk", "missing_pair", "orig_done", "sem_done", "state_hash_before_equal",
        "orig_no_commit", "sem_no_commit", "probe_no_commit_exact", "orig_rows", "sem_rows",
        "orig_mean_D", "sem_mean_D", "orig_q90_D", "sem_q90_D", "orig_mass_D_gt_050",
        "sem_mass_D_gt_050", "sem_label_count_mean", "sem_fallback_mean", "sem_applied_rows",
        "orig_pose_t_mean", "sem_pose_t_mean", "orig_pose_t_max", "sem_pose_t_max",
        "orig_pose_r_deg_mean", "sem_pose_r_deg_mean", "hash_H_next_equals_before",
        "D_delta_rms_proxy", "delta_mean_D", "delta_q90_D", "delta_mass_D_gt_050",
        "delta_semantic_d_mean", "delta_pose_t_mean", "delta_pose_t_max",
        "delta_pose_r_deg_mean", "oracle_positive_diagnostic",
    ])
    write_csv(out_dir / "paired_probe_decisions.csv", decisions, [
        "parent", "chunk", "probe_no_commit_exact", "state_hash_before_equal", "label_ok",
        "D_delta_rms_proxy", "delta_q90_D", "delta_mass_D_gt_050", "improve_count",
        "delta_pose_t_mean", "delta_pose_t_max", "delta_pose_r_deg_mean",
        "hard_regress_count", "G_probe", "oracle_positive_diagnostic",
    ])
    write_csv(out_dir / "probe_state_hash_audit.csv", decisions, [
        "parent", "chunk", "orig_no_commit", "sem_no_commit", "probe_no_commit_exact",
        "state_hash_before_equal", "hash_H_next_equals_before",
    ])
    (out_dir / "paired_probe_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0 if summary["gate_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
