#!/usr/bin/env python3
"""Merge v119TF LB-LOGICAL TR component-ablation roots into one audit summary."""

from __future__ import annotations

import csv
import json
import math
import os
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "results/acl2_v119tf_semantic_addressable_geometry_carrier_routing_carrier_aware_augmented"
BASE_RUN_ROOT = RESULT_ROOT / os.environ.get(
    "ACL2_V119_LB_COMPONENT_ABLATION_BASE_ROOT",
    "stage2_lblogical_tr_semantic_ablation",
).strip()
CONTROL_RUN_ROOT = RESULT_ROOT / os.environ.get(
    "ACL2_V119_LB_COMPONENT_ABLATION_CONTROL_ROOT",
    "stage2_lblogical_tr_semantic_controls",
).strip()
BASE_SUMMARY_CSV = BASE_RUN_ROOT / os.environ.get(
    "ACL2_V119_LB_COMPONENT_ABLATION_BASE_CSV",
    "lblogical_tr_semantic_ablation_summary_seq00.csv",
).strip()
SEQ = os.environ.get("ACL2_V119_LB_STAGE2_SEQ", "00").strip() or "00"
CONTROL_DATASET_TAG = os.environ.get(
    "ACL2_V119_LB_COMPONENT_ABLATION_CONTROL_DATASET_TAG",
    "lblogical_tr_semantic_controls",
).strip()
CONTROL_DATASET = f"kitti_v119_{CONTROL_DATASET_TAG}_seq{SEQ}"
SUMMARY_STEM = os.environ.get(
    "ACL2_V119_LB_COMPONENT_ABLATION_SUMMARY_STEM",
    "lblogical_tr_component_ablation_summary_seq00",
).strip()
CANDIDATE_VARIANT = "tr5_logical_stable_qk_topk"
COMPARISON_VARIANTS = [
    "tr0_default_no_policy",
    "tr1_page_qk_topk",
    "tr8_logical_internal_qk_topk2",
    "tr9_logical_provenance_shuffle_qk_topk2",
    "tr10_logical_role_swap_qk_topk2",
    "tr6_logical_stable_reverse_qk",
    "tr7_logical_stable_random_seed00",
]


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def finite_float(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def pct_delta(lower_is_better_value: Any, baseline_value: Any) -> float | str:
    if not finite_float(lower_is_better_value) or not finite_float(baseline_value):
        return ""
    value = float(lower_is_better_value)
    baseline = float(baseline_value)
    if baseline == 0:
        return ""
    return 100.0 * (baseline - value) / baseline


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def log_has_markers(path: Path, markers: list[str]) -> bool:
    if not path.exists():
        return False
    text = path.read_text(encoding="utf-8", errors="replace")
    return all(marker in text for marker in markers)


def action_line_count(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def fi_semantic_control_counts(path: Path) -> str:
    counts = Counter()
    if path.exists():
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if row.get("operation_type") == "stage4_logical_retrieval_policy_select":
                    counts[str(row.get("logical_semantic_control", ""))] += 1
    return json.dumps(dict(sorted(counts.items())), sort_keys=True)


def control_rows() -> list[dict[str, Any]]:
    manifest_path = CONTROL_RUN_ROOT / "run_manifest.csv"
    metrics_path = CONTROL_RUN_ROOT / "workspace" / CONTROL_DATASET / "eval" / "traj.json"
    metrics_by_method = json.loads(metrics_path.read_text(encoding="utf-8"))
    rows = []
    for manifest in read_csv(manifest_path):
        if manifest.get("phase") != "run_worker":
            continue
        variant = str(manifest["variant"])
        method = str(manifest["method"])
        action_path = Path(str(manifest["action_file"]))
        fi_path = Path(str(manifest["fi_trace"]))
        run_log = Path(str(manifest["log"]))
        eval_log = CONTROL_RUN_ROOT / "logs" / f"evaluate_{variant}_seq{SEQ}.log"
        method_root = CONTROL_RUN_ROOT / "workspace" / CONTROL_DATASET / SEQ / method
        metrics = metrics_by_method.get(method, {})
        rows.append(
            {
                "seq": SEQ,
                "dataset": CONTROL_DATASET,
                "variant": variant,
                "role": manifest.get("role", ""),
                "policy": manifest.get("policy", ""),
                "role_filter": manifest.get("role_filter", ""),
                "min_semantic_score": manifest.get("min_semantic_score", ""),
                "semantic_control": manifest.get("semantic_control", ""),
                "method": method,
                "run_root": rel(CONTROL_RUN_ROOT),
                "worker_log": rel(run_log),
                "evaluate_log": rel(eval_log),
                "action_file": rel(action_path),
                "fi_trace": rel(fi_path),
                "ate": metrics.get("ate", ""),
                "rpe_rot": metrics.get("rpe_rot", ""),
                "rpe_trans": metrics.get("rpe_trans", ""),
                "num_scenes": metrics.get("num_scenes", ""),
                "raw_action_row_count": action_line_count(action_path),
                "fi_trace_row_count": action_line_count(fi_path),
                "logical_semantic_controls": fi_semantic_control_counts(fi_path),
                "complete_exists": int((method_root / ".complete.json").exists()),
                "traj_exists": int((method_root / "traj.txt").exists()),
                "worker_log_success_markers": log_has_markers(
                    run_log,
                    ["Completed successfully", "Worker done: 1/1 scenes succeeded"],
                ),
                "evaluate_log_success_markers": log_has_markers(
                    eval_log,
                    ["Total successful: 1", "Total failed: 0"],
                ),
            }
        )
    return rows


def main() -> None:
    base_rows = read_csv(BASE_SUMMARY_CSV)
    for row in base_rows:
        row.setdefault("run_root", rel(BASE_RUN_ROOT))
    rows = base_rows + control_rows()
    by_variant = {str(row["variant"]): row for row in rows}
    candidate = by_variant[CANDIDATE_VARIANT]
    for variant, row in by_variant.items():
        row["ate_improvement_pct_vs_candidate"] = pct_delta(row.get("ate"), candidate.get("ate"))
        row["candidate_ate_improvement_pct_vs_this"] = pct_delta(candidate.get("ate"), row.get("ate"))
        row["candidate_beats_this"] = bool(
            finite_float(candidate.get("ate"))
            and finite_float(row.get("ate"))
            and float(candidate["ate"]) < float(row["ate"])
        )
    comparisons = {
        variant: {
            "ate": by_variant.get(variant, {}).get("ate", ""),
            "candidate_ate_improvement_pct_vs_this": by_variant.get(variant, {}).get(
                "candidate_ate_improvement_pct_vs_this", ""
            ),
            "candidate_beats_this": by_variant.get(variant, {}).get("candidate_beats_this", ""),
        }
        for variant in COMPARISON_VARIANTS
        if variant in by_variant
    }
    terminal_pass = bool(
        finite_float(candidate.get("ate"))
        and all(
            variant == CANDIDATE_VARIANT
            or (
                finite_float(by_variant.get(variant, {}).get("ate"))
                and float(candidate["ate"]) < float(by_variant[variant]["ate"])
            )
            for variant in COMPARISON_VARIANTS
            if variant in by_variant
        )
        and finite_float(candidate.get("ate_improvement_pct_vs_default"))
        and float(candidate["ate_improvement_pct_vs_default"]) >= 1.0
    )
    summary_csv = RESULT_ROOT / f"{SUMMARY_STEM}.csv"
    summary_json = RESULT_ROOT / f"{SUMMARY_STEM}.json"
    write_csv(summary_csv, rows)
    summary = {
        "schema": "acl2_v119tf_lblogical_tr_component_ablation_summary_v1",
        "base_summary_csv": rel(BASE_SUMMARY_CSV),
        "control_run_root": rel(CONTROL_RUN_ROOT),
        "summary_csv": rel(summary_csv),
        "summary_json": rel(summary_json),
        "candidate_variant": CANDIDATE_VARIANT,
        "candidate_ate": candidate.get("ate", ""),
        "comparison_variants": comparisons,
        "component_ablation_terminal_pass": terminal_pass,
        "truthfulness_boundary": (
            "seq00 LB-LOGICAL TR component ablation only; no holdout, dense, scale, "
            "or full v119 branch completion is claimed."
        ),
    }
    summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
