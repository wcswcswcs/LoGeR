#!/usr/bin/env python3
"""Summarize ACL2 v119-TF Stage2 LB-LOGICAL TR pilot runtime artifacts."""

from __future__ import annotations

import csv
import json
import math
import os
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "results/acl2_v119tf_semantic_addressable_geometry_carrier_routing_carrier_aware_augmented"
RUN_ROOT_NAME = os.environ.get("ACL2_V119_LB_STAGE2_RUN_ROOT_NAME", "stage2_lblogical_tr_pilot").strip()
RUN_ROOT = RESULT_ROOT / (RUN_ROOT_NAME or "stage2_lblogical_tr_pilot")
WORKSPACE = RUN_ROOT / "workspace"
MANIFEST = RUN_ROOT / "run_manifest.csv"
SEQ = os.environ.get("ACL2_V119_LB_STAGE2_SEQ", "00").strip() or "00"
DATASET_TAG = os.environ.get("ACL2_V119_LB_STAGE2_DATASET_TAG", "lblogical_tr_pilot").strip() or "lblogical_tr_pilot"
DATASET = f"kitti_v119_{DATASET_TAG}_seq{SEQ}"
NUM_FRAMES = 4541
SCALE_FRAMES = 8
EXPECTED_ACTION_ROWS = NUM_FRAMES - SCALE_FRAMES
CANDIDATE_VARIANT = os.environ.get("ACL2_V119_LB_STAGE2_CANDIDATE_VARIANT", "tr2_logical_qk_topk").strip()
DEFAULT_VARIANT = os.environ.get("ACL2_V119_LB_STAGE2_DEFAULT_VARIANT", "tr0_default_no_policy").strip()
PAGE_CONTROL_VARIANT = os.environ.get("ACL2_V119_LB_STAGE2_PAGE_CONTROL_VARIANT", "tr1_page_qk_topk").strip()
REVERSE_CONTROL_VARIANT = os.environ.get("ACL2_V119_LB_STAGE2_REVERSE_CONTROL_VARIANT", "tr3_logical_reverse_qk").strip()
RANDOM_CONTROL_VARIANT = os.environ.get("ACL2_V119_LB_STAGE2_RANDOM_CONTROL_VARIANT", "tr4_logical_random_seed00").strip()
INTERNAL_ABLATION_VARIANT = os.environ.get("ACL2_V119_LB_STAGE2_INTERNAL_ABLATION_VARIANT", "").strip()
SUMMARY_STEM = os.environ.get(
    "ACL2_V119_LB_STAGE2_SUMMARY_STEM",
    f"lblogical_tr_pilot_summary_seq{SEQ}",
).strip() or f"lblogical_tr_pilot_summary_seq{SEQ}"


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def finite_float(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def mean(values: list[float]) -> float | str:
    return float(statistics.fmean(values)) if values else ""


def median(values: list[float]) -> float | str:
    return float(statistics.median(values)) if values else ""


def pct_delta(lower_is_better_value: Any, baseline_value: Any) -> float | str:
    if not finite_float(lower_is_better_value) or not finite_float(baseline_value):
        return ""
    value = float(lower_is_better_value)
    baseline = float(baseline_value)
    if baseline == 0:
        return ""
    return 100.0 * (baseline - value) / baseline


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


def read_manifest() -> dict[str, dict[str, str]]:
    rows: dict[str, dict[str, str]] = {}
    with MANIFEST.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["phase"] == "run_worker":
                rows[row["variant"]] = row
    return rows


def load_metrics() -> dict[str, dict[str, Any]]:
    path = WORKSPACE / DATASET / "eval" / "traj.json"
    return json.loads(path.read_text(encoding="utf-8"))


def log_has_markers(path: Path, markers: list[str]) -> bool:
    if not path.exists():
        return False
    text = path.read_text(encoding="utf-8", errors="replace")
    return all(marker in text for marker in markers)


def action_summary(path: Path) -> dict[str, Any]:
    count = 0
    positions: list[int] = []
    modes = Counter()
    skips = Counter()
    final_keyframes = 0
    base_keyframes = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            count += 1
            if "sample_position" in row:
                positions.append(int(row["sample_position"]))
            modes[str(row.get("keyframe_schedule_mode", ""))] += 1
            skips[str(bool(row.get("skip_append")))] += 1
            final_keyframes += int(bool(row.get("final_is_keyframe")))
            base_keyframes += int(bool(row.get("base_is_keyframe")))
    return {
        "raw_action_row_count": count,
        "expected_action_row_count": EXPECTED_ACTION_ROWS,
        "raw_action_first_sample_position": min(positions) if positions else "",
        "raw_action_last_sample_position": max(positions) if positions else "",
        "raw_action_unique_sample_position_count": len(set(positions)),
        "raw_action_duplicate_sample_position_count": count - len(set(positions)),
        "raw_action_missing_sample_position_count": len(set(range(SCALE_FRAMES, NUM_FRAMES)) - set(positions)),
        "raw_action_extra_sample_position_count": len(set(positions) - set(range(SCALE_FRAMES, NUM_FRAMES))),
        "raw_action_schedule_modes": json.dumps(dict(sorted(modes.items())), sort_keys=True),
        "raw_action_skip_append_counts": json.dumps(dict(sorted(skips.items())), sort_keys=True),
        "raw_action_final_keyframe_count": final_keyframes,
        "raw_action_base_keyframe_count": base_keyframes,
        "raw_action_exact_frame_coverage_pass": bool(
            count == EXPECTED_ACTION_ROWS
            and len(set(positions)) == EXPECTED_ACTION_ROWS
            and not (set(range(SCALE_FRAMES, NUM_FRAMES)) - set(positions))
        ),
    }


def parse_semantic_payload(value: Any) -> dict[str, Any]:
    if not value:
        return {}
    try:
        payload = json.loads(str(value))
        return payload if isinstance(payload, dict) else {}
    except json.JSONDecodeError:
        return {}


def fi_summary(path: Path) -> dict[str, Any]:
    row_count = 0
    operation_counts = Counter()
    logical_action_count = 0
    logical_read_count = 0
    selected_counts: list[float] = []
    logical_entry_counts: list[float] = []
    mask_true_counts: list[float] = []
    mask_total_counts: list[float] = []
    selected_special_page_counts: list[float] = []
    stage4_policies = Counter()
    semantic_controls = Counter()
    logical_backends = Counter()
    read_roles = Counter()
    read_labels = Counter()
    read_frames: set[int] = set()
    read_entries: set[str] = set()
    semantic_scores: list[float] = []
    qk_scores: list[float] = []
    read_entropies: list[float] = []
    action_frames: set[int] = set()

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            row_count += 1
            op = str(row.get("operation_type", ""))
            operation_counts[op] += 1
            if op == "stage4_logical_retrieval_policy_select":
                logical_action_count += 1
                stage4_policies[str(row.get("stage4_policy", ""))] += 1
                semantic_controls[str(row.get("logical_semantic_control", ""))] += 1
                if finite_float(row.get("selected_logical_entry_count")):
                    selected_counts.append(float(row["selected_logical_entry_count"]))
                if finite_float(row.get("logical_entry_count")):
                    logical_entry_counts.append(float(row["logical_entry_count"]))
                if finite_float(row.get("custom_mask_true_count")):
                    mask_true_counts.append(float(row["custom_mask_true_count"]))
                if finite_float(row.get("custom_mask_total_count")):
                    mask_total_counts.append(float(row["custom_mask_total_count"]))
                if finite_float(row.get("selected_special_page_count")):
                    selected_special_page_counts.append(float(row["selected_special_page_count"]))
                if "current_frame" in row:
                    action_frames.add(int(row["current_frame"]))
            elif op == "runtime_read_subrange_mask_logical_special_entry":
                logical_read_count += 1
                logical_backends[str(row.get("logical_backend", ""))] += 1
                if "source_frame_id" in row:
                    read_frames.add(int(row["source_frame_id"]))
                if row.get("logical_entry_id"):
                    read_entries.add(str(row["logical_entry_id"]))
                if finite_float(row.get("entry_qk_score")):
                    qk_scores.append(float(row["entry_qk_score"]))
                if finite_float(row.get("entry_read_entropy")):
                    read_entropies.append(float(row["entry_read_entropy"]))
                sem = parse_semantic_payload(row.get("entry_semantic_provenance"))
                if sem:
                    read_roles[str(sem.get("best_track_role", ""))] += 1
                    read_labels[str(sem.get("best_track_label", ""))] += 1
                    if finite_float(sem.get("score")):
                        semantic_scores.append(float(sem["score"]))

    mask_false_counts = [total - true for total, true in zip(mask_total_counts, mask_true_counts)]
    mask_true_fraction = [
        true / total
        for true, total in zip(mask_true_counts, mask_total_counts)
        if total
    ]
    return {
        "fi_trace_row_count": row_count,
        "fi_trace_operation_counts": json.dumps(dict(sorted(operation_counts.items())), sort_keys=True),
        "logical_policy_action_count": logical_action_count,
        "logical_policy_action_frame_count": len(action_frames),
        "logical_read_row_count": logical_read_count,
        "logical_read_unique_source_frame_count": len(read_frames),
        "logical_read_unique_entry_count": len(read_entries),
        "logical_stage4_policies": json.dumps(dict(sorted(stage4_policies.items())), sort_keys=True),
        "logical_semantic_controls": json.dumps(dict(sorted(semantic_controls.items())), sort_keys=True),
        "logical_backends": json.dumps(dict(sorted(logical_backends.items())), sort_keys=True),
        "logical_selected_entry_count_mean": mean(selected_counts),
        "logical_selected_entry_count_min": min(selected_counts) if selected_counts else "",
        "logical_selected_entry_count_max": max(selected_counts) if selected_counts else "",
        "logical_entry_count_mean": mean(logical_entry_counts),
        "logical_custom_mask_true_count_mean": mean(mask_true_counts),
        "logical_custom_mask_false_count_mean": mean(mask_false_counts),
        "logical_custom_mask_true_fraction_mean": mean(mask_true_fraction),
        "logical_selected_special_page_count_mean": mean(selected_special_page_counts),
        "logical_read_semantic_roles": json.dumps(dict(sorted(read_roles.items())), sort_keys=True),
        "logical_read_semantic_labels": json.dumps(dict(sorted(read_labels.items())), sort_keys=True),
        "logical_read_semantic_score_mean": mean(semantic_scores),
        "logical_read_semantic_score_median": median(semantic_scores),
        "logical_read_qk_score_mean": mean(qk_scores),
        "logical_read_qk_score_median": median(qk_scores),
        "logical_read_entropy_mean": mean(read_entropies),
        "runtime_logical_evidence_pass": bool(logical_action_count > 0 and logical_read_count > 0),
    }


def summarize() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest_rows = read_manifest()
    metrics_by_method = load_metrics()
    rows: list[dict[str, Any]] = []
    for variant, manifest in manifest_rows.items():
        method = manifest["method"]
        action_path = Path(manifest["action_file"])
        fi_path = Path(manifest["fi_trace"])
        run_log = Path(manifest["log"])
        eval_log = RUN_ROOT / "logs" / f"evaluate_{variant}_seq{SEQ}.log"
        method_root = WORKSPACE / DATASET / SEQ / method
        metrics = metrics_by_method.get(method, {})
        row: dict[str, Any] = {
            "seq": SEQ,
            "dataset": DATASET,
            "variant": variant,
            "role": manifest.get("role", ""),
            "policy": manifest.get("policy", ""),
            "role_filter": manifest.get("role_filter", ""),
            "min_semantic_score": manifest.get("min_semantic_score", ""),
            "semantic_control": manifest.get("semantic_control", ""),
            "method": method,
            "gpu": manifest.get("gpu", ""),
            "worker_log": rel(run_log),
            "evaluate_log": rel(eval_log),
            "action_file": rel(action_path),
            "fi_trace": rel(fi_path),
            "complete_exists": int((method_root / ".complete.json").exists()),
            "traj_exists": int((method_root / "traj.txt").exists()),
            "intrinsics_exists": int((method_root / "intrinsics.txt").exists()),
            "depth_exr_count": len(list((method_root / "depth").glob("*.exr"))),
            "confidence_exr_count": len(list((method_root / "confidence").glob("*.exr"))),
            "ate": metrics.get("ate", ""),
            "rpe_rot": metrics.get("rpe_rot", ""),
            "rpe_trans": metrics.get("rpe_trans", ""),
            "num_scenes": metrics.get("num_scenes", ""),
            "metrics_present": all(finite_float(metrics.get(key)) for key in ("ate", "rpe_rot", "rpe_trans")),
            "worker_log_success_markers": log_has_markers(run_log, ["Completed successfully", "Worker done: 1/1 scenes succeeded"]),
            "evaluate_log_success_markers": log_has_markers(eval_log, ["Total successful: 1", "Total failed: 0"]),
        }
        row.update(action_summary(action_path))
        row.update(fi_summary(fi_path))
        row["runtime_pilot_row_pass"] = bool(
            row["metrics_present"]
            and row["worker_log_success_markers"]
            and row["evaluate_log_success_markers"]
            and row["raw_action_exact_frame_coverage_pass"]
            and row["complete_exists"]
            and row["traj_exists"]
            and row["intrinsics_exists"]
        )
        rows.append(row)

    by_variant = {row["variant"]: row for row in rows}
    default = by_variant[DEFAULT_VARIANT]
    page = by_variant[PAGE_CONTROL_VARIANT]
    random = by_variant[RANDOM_CONTROL_VARIANT]
    reverse = by_variant[REVERSE_CONTROL_VARIANT]
    candidate = by_variant[CANDIDATE_VARIANT]
    internal_ablation = None
    if INTERNAL_ABLATION_VARIANT:
        if INTERNAL_ABLATION_VARIANT not in by_variant:
            raise KeyError(
                f"internal ablation variant {INTERNAL_ABLATION_VARIANT!r} missing from manifest rows"
            )
        internal_ablation = by_variant[INTERNAL_ABLATION_VARIANT]
    for row in rows:
        row["ate_improvement_pct_vs_default"] = pct_delta(row.get("ate"), default.get("ate"))
        row["rpe_trans_improvement_pct_vs_default"] = pct_delta(row.get("rpe_trans"), default.get("rpe_trans"))
        row["ate_improvement_pct_vs_page_control"] = pct_delta(row.get("ate"), page.get("ate"))
        row["ate_improvement_pct_vs_random_control"] = pct_delta(row.get("ate"), random.get("ate"))
        row["ate_improvement_pct_vs_reverse_control"] = pct_delta(row.get("ate"), reverse.get("ate"))
        if internal_ablation is not None:
            row["ate_improvement_pct_vs_internal_ablation"] = pct_delta(
                row.get("ate"), internal_ablation.get("ate")
            )
    candidate_geometry_pass_vs_default_1pct = bool(
        finite_float(candidate.get("ate_improvement_pct_vs_default"))
        and float(candidate["ate_improvement_pct_vs_default"]) >= 1.0
    )
    candidate_beats_random = bool(
        finite_float(candidate.get("ate"))
        and finite_float(random.get("ate"))
        and float(candidate["ate"]) < float(random["ate"])
    )
    candidate_beats_reverse = bool(
        finite_float(candidate.get("ate"))
        and finite_float(reverse.get("ate"))
        and float(candidate["ate"]) < float(reverse["ate"])
    )
    candidate_beats_page = bool(
        finite_float(candidate.get("ate"))
        and finite_float(page.get("ate"))
        and float(candidate["ate"]) < float(page["ate"])
    )
    candidate_beats_internal_ablation = (
        bool(
            finite_float(candidate.get("ate"))
            and finite_float(internal_ablation.get("ate"))
            and float(candidate["ate"]) < float(internal_ablation["ate"])
        )
        if internal_ablation is not None
        else ""
    )
    candidate_evidence_pass = bool(candidate["runtime_logical_evidence_pass"])
    candidate_runtime_pass = bool(candidate["runtime_pilot_row_pass"])
    carrier_route_terminal_pass = bool(
        candidate_runtime_pass
        and candidate_evidence_pass
        and candidate_geometry_pass_vs_default_1pct
        and candidate_beats_random
        and candidate_beats_reverse
        and candidate_beats_page
    )
    summary = {
        "schema": "acl2_v119tf_stage2_lblogical_tr_pilot_summary_v1",
        "run_root": rel(RUN_ROOT),
        "seq": SEQ,
        "dataset": DATASET,
        "dataset_tag": DATASET_TAG,
        "num_frames": NUM_FRAMES,
        "scale_frames": SCALE_FRAMES,
        "expected_action_rows": EXPECTED_ACTION_ROWS,
        "variant_count": len(rows),
        "runtime_worker_eval_all_pass": all(bool(row["runtime_pilot_row_pass"]) for row in rows),
        "candidate_variant": CANDIDATE_VARIANT,
        "candidate_ate": candidate.get("ate"),
        "default_ate": default.get("ate"),
        "page_control_ate": page.get("ate"),
        "reverse_control_ate": reverse.get("ate"),
        "random_control_ate": random.get("ate"),
        "internal_ablation_variant": INTERNAL_ABLATION_VARIANT,
        "internal_ablation_ate": internal_ablation.get("ate") if internal_ablation is not None else "",
        "candidate_ate_improvement_pct_vs_default": candidate.get("ate_improvement_pct_vs_default"),
        "candidate_ate_improvement_pct_vs_page_control": candidate.get("ate_improvement_pct_vs_page_control"),
        "candidate_ate_improvement_pct_vs_random_control": candidate.get("ate_improvement_pct_vs_random_control"),
        "candidate_ate_improvement_pct_vs_reverse_control": candidate.get("ate_improvement_pct_vs_reverse_control"),
        "candidate_ate_improvement_pct_vs_internal_ablation": (
            candidate.get("ate_improvement_pct_vs_internal_ablation")
            if internal_ablation is not None
            else ""
        ),
        "candidate_runtime_logical_evidence_pass": candidate_evidence_pass,
        "candidate_geometry_pass_vs_default_1pct": candidate_geometry_pass_vs_default_1pct,
        "candidate_beats_page_control": candidate_beats_page,
        "candidate_beats_random_control": candidate_beats_random,
        "candidate_beats_reverse_control": candidate_beats_reverse,
        "candidate_beats_internal_ablation": candidate_beats_internal_ablation,
        "carrier_route_terminal_pass": carrier_route_terminal_pass,
        "truthfulness_boundary": (
            "seq00 LB-LOGICAL TR pilot only. This is not full v119 success, does not cover all "
            "carrier branches, and does not satisfy final dense/scale/local gates."
        ),
    }
    return rows, summary


def main() -> None:
    rows, summary = summarize()
    summary_csv = RUN_ROOT / f"{SUMMARY_STEM}.csv"
    summary_json = RUN_ROOT / f"{SUMMARY_STEM}.json"
    carrier_csv = RESULT_ROOT / "V119_CARRIER_EVIDENCE_ROWS.csv"
    carrier_parquet = RESULT_ROOT / "V119_CARRIER_EVIDENCE_ROWS.parquet"
    write_csv(summary_csv, rows)
    write_csv(carrier_csv, rows)
    pd.DataFrame(rows).replace({"": None}).to_parquet(carrier_parquet, index=False)
    summary.update(
        {
            "summary_csv": rel(summary_csv),
            "summary_json": rel(summary_json),
            "carrier_evidence_csv": rel(carrier_csv),
            "carrier_evidence_parquet": rel(carrier_parquet),
            "carrier_evidence_row_count": len(rows),
        }
    )
    summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
