#!/usr/bin/env python3
"""Summarize v119TF Track C C-LB-P minimum-mechanism runtime artifacts."""

from __future__ import annotations

import csv
import json
import math
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "results/acl2_v119tf_semantic_addressable_geometry_carrier_routing_carrier_aware_augmented"
MINIMECH_LABEL = (
    os.environ.get("ACL2_V119_MINIMECH_LABEL")
    or os.environ.get("ACL2_V119_CLBP_LABEL")
    or "clbp"
).strip().lower()
EXPLICIT_BRANCH = (
    os.environ.get("ACL2_V119_MINIMECH_EXPLICIT_BRANCH")
    or os.environ.get("ACL2_V119_CLBP_EXPLICIT_BRANCH")
    or "C-LB-P"
).strip()
RUN_ROOT_NAME = (
    os.environ.get("ACL2_V119_MINIMECH_RUN_ROOT_NAME")
    or os.environ.get("ACL2_V119_CLBP_RUN_ROOT_NAME")
    or "stage3_clbp_minimech"
).strip()
RUN_ROOT = RESULT_ROOT / (RUN_ROOT_NAME or "stage3_clbp_minimech")
WORKSPACE = RUN_ROOT / "workspace"
MANIFEST = RUN_ROOT / "run_manifest.csv"
SEQ = (os.environ.get("ACL2_V119_MINIMECH_SEQ") or os.environ.get("ACL2_V119_CLBP_SEQ") or "00").strip() or "00"
DATASET_TAG = (
    os.environ.get("ACL2_V119_MINIMECH_DATASET_TAG")
    or os.environ.get("ACL2_V119_CLBP_DATASET_TAG")
    or "clbp_minimech"
).strip() or "clbp_minimech"
DATASET = f"kitti_v119_{DATASET_TAG}_seq{SEQ}"
NUM_FRAMES = 4541
SCALE_FRAMES = 8
EXPECTED_ACTION_ROWS = NUM_FRAMES - SCALE_FRAMES
SUMMARY_STEM = (
    os.environ.get("ACL2_V119_MINIMECH_SUMMARY_STEM")
    or os.environ.get("ACL2_V119_CLBP_SUMMARY_STEM")
    or f"{MINIMECH_LABEL}_minimech_summary_seq{SEQ}"
).strip()
ROOT_EVIDENCE_NAME = (
    os.environ.get("ACL2_V119_MINIMECH_ROOT_EVIDENCE_NAME")
    or f"V119_{MINIMECH_LABEL.upper()}_MINIMECH_EVIDENCE_ROWS.csv"
).strip()
REQUIRED_FORM_COUNT = int(os.environ.get("ACL2_V119_MINIMECH_REQUIRED_FORM_COUNT", "4"))
REQUIRED_CONTROL_COUNT = int(os.environ.get("ACL2_V119_MINIMECH_REQUIRED_CONTROL_COUNT", "7"))
WORKER_SUCCESS_MARKERS = ["Completed successfully", "Worker done: 1/1 scenes succeeded"]
EVALUATE_SUCCESS_MARKERS = ["Total successful: 1", "Total failed: 0"]


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def finite_float(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def pct_delta(value: Any, baseline: Any) -> float | str:
    if not finite_float(value) or not finite_float(baseline) or float(baseline) == 0.0:
        return ""
    return 100.0 * (float(baseline) - float(value)) / float(baseline)


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


def read_manifest() -> list[dict[str, str]]:
    with MANIFEST.open("r", encoding="utf-8", newline="") as handle:
        return [row for row in csv.DictReader(handle) if row.get("phase") == "run_worker"]


def load_metrics() -> dict[str, dict[str, Any]]:
    path = WORKSPACE / DATASET / "eval" / "traj.json"
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def log_has_markers(path: Path, markers: list[str]) -> bool:
    if not path.is_file():
        return False
    text = path.read_text(encoding="utf-8", errors="replace")
    return all(marker in text for marker in markers)


def resolved_worker_log(path: Path) -> tuple[Path, bool]:
    if log_has_markers(path, WORKER_SUCCESS_MARKERS):
        return path, False
    for suffix in ("_retry3.log", "_retry2.log", "_retry1.log"):
        retry = path.with_name(path.stem + suffix)
        if log_has_markers(retry, WORKER_SUCCESS_MARKERS):
            return retry, True
    for candidate in sorted(path.parent.glob(path.stem + "*threadcap*.log")):
        if log_has_markers(candidate, WORKER_SUCCESS_MARKERS):
            return candidate, True
    gpu_stem = re.sub(r"_gpu\d+$", "_gpu", path.stem)
    if gpu_stem != path.stem:
        for candidate in sorted(path.parent.glob(gpu_stem + "*threadcap*.log")):
            if log_has_markers(candidate, WORKER_SUCCESS_MARKERS):
                return candidate, True
    return path, False


def action_summary(path: Path) -> dict[str, Any]:
    count = 0
    positions: set[int] = set()
    if path.is_file():
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                count += 1
                if "sample_position" in row:
                    positions.add(int(row["sample_position"]))
    expected_positions = set(range(SCALE_FRAMES, NUM_FRAMES))
    return {
        "raw_action_row_count": count,
        "expected_action_row_count": EXPECTED_ACTION_ROWS,
        "raw_action_unique_sample_position_count": len(positions),
        "raw_action_missing_sample_position_count": len(expected_positions - positions),
        "raw_action_extra_sample_position_count": len(positions - expected_positions),
        "raw_action_exact_frame_coverage_pass": bool(
            count == EXPECTED_ACTION_ROWS
            and len(positions) == EXPECTED_ACTION_ROWS
            and not (expected_positions - positions)
        ),
    }


def parse_semantic(value: Any) -> dict[str, Any]:
    if not value:
        return {}
    try:
        parsed = json.loads(str(value))
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def fi_summary(path: Path) -> dict[str, Any]:
    ops = Counter()
    branches = Counter()
    admission_modes = Counter()
    semantic_controls = Counter()
    roles = Counter()
    selected_actual_roles = Counter()
    selected_policy_roles = Counter()
    selected_local_lanes = Counter()
    read_local_lanes = Counter()
    anchor_relevance_scores: list[float] = []
    metric_qualification_scores: list[float] = []
    logical_select = 0
    logical_read = 0
    filter_abstain = 0
    retained_counts: list[int] = []
    rows = 0
    if path.is_file():
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                rows += 1
                op = str(row.get("operation_type", ""))
                ops[op] += 1
                if row.get("explicit_carrier_branch"):
                    branches[str(row.get("explicit_carrier_branch"))] += 1
                if row.get("logical_admission_mode"):
                    admission_modes[str(row.get("logical_admission_mode"))] += 1
                if "logical_semantic_control" in row:
                    semantic_controls[str(row.get("logical_semantic_control", ""))] += 1
                if row.get("local_lane"):
                    read_local_lanes[str(row.get("local_lane", ""))] += 1
                if op == "stage4_logical_retrieval_policy_select":
                    logical_select += 1
                    if finite_float(row.get("logical_retained_entry_count")):
                        retained_counts.append(int(float(row["logical_retained_entry_count"])))
                    for role in str(row.get("selected_semantic_roles_preview", "")).split(","):
                        if role:
                            selected_policy_roles[role] += 1
                    for role in str(row.get("selected_actual_semantic_roles_preview", "")).split(","):
                        if role:
                            selected_actual_roles[role] += 1
                    for lane in str(row.get("selected_local_lanes_preview", "")).split(","):
                        if lane:
                            selected_local_lanes[lane] += 1
                    for score in str(row.get("selected_anchor_relevance_scores_preview", "")).split(","):
                        if finite_float(score):
                            anchor_relevance_scores.append(float(score))
                    for score in str(row.get("selected_metric_qualification_scores_preview", "")).split(","):
                        if finite_float(score):
                            metric_qualification_scores.append(float(score))
                elif op == "stage4_logical_retrieval_policy_filter_abstain":
                    filter_abstain += 1
                elif op == "runtime_read_subrange_mask_logical_special_entry":
                    logical_read += 1
                    sem = parse_semantic(row.get("entry_semantic_provenance"))
                    if sem:
                        roles[str(sem.get("best_track_role", ""))] += 1
    return {
        "fi_trace_row_count": rows,
        "fi_trace_operation_counts": json.dumps(dict(sorted(ops.items())), sort_keys=True),
        "explicit_carrier_branch_counts": json.dumps(dict(sorted(branches.items())), sort_keys=True),
        "logical_admission_mode_counts": json.dumps(dict(sorted(admission_modes.items())), sort_keys=True),
        "logical_semantic_control_counts": json.dumps(dict(sorted(semantic_controls.items())), sort_keys=True),
        "logical_policy_select_count": logical_select,
        "logical_policy_filter_abstain_count": filter_abstain,
        "logical_read_row_count": logical_read,
        "logical_retained_entry_count_min": min(retained_counts) if retained_counts else "",
        "logical_retained_entry_count_max": max(retained_counts) if retained_counts else "",
        "logical_read_semantic_roles": json.dumps(dict(sorted(roles.items())), sort_keys=True),
        "selected_policy_roles": json.dumps(dict(sorted(selected_policy_roles.items())), sort_keys=True),
        "selected_actual_roles": json.dumps(dict(sorted(selected_actual_roles.items())), sort_keys=True),
        "selected_local_lanes": json.dumps(dict(sorted(selected_local_lanes.items())), sort_keys=True),
        "read_local_lanes": json.dumps(dict(sorted(read_local_lanes.items())), sort_keys=True),
        "selected_anchor_relevance_score_min": min(anchor_relevance_scores) if anchor_relevance_scores else "",
        "selected_anchor_relevance_score_max": max(anchor_relevance_scores) if anchor_relevance_scores else "",
        "selected_metric_qualification_score_min": min(metric_qualification_scores) if metric_qualification_scores else "",
        "selected_metric_qualification_score_max": max(metric_qualification_scores) if metric_qualification_scores else "",
        "runtime_explicit_evidence_pass": bool(
            logical_select > 0
            and logical_read > 0
            and branches.get(EXPLICIT_BRANCH, 0) > 0
        ),
        "runtime_clbp_evidence_pass": bool(logical_select > 0 and logical_read > 0 and branches.get("C-LB-P", 0) > 0),
    }


def build_rows() -> list[dict[str, Any]]:
    metrics = load_metrics()
    rows: list[dict[str, Any]] = []
    for manifest in read_manifest():
        variant = manifest["variant"]
        method = manifest["method"]
        method_root = WORKSPACE / DATASET / SEQ / method
        eval_log = RUN_ROOT / "logs" / f"evaluate_{variant}_seq{SEQ}.log"
        worker_log, worker_log_retry_used = resolved_worker_log(Path(manifest["log"]))
        row = {
            "seq": SEQ,
            "dataset": DATASET,
            "variant": variant,
            "role": manifest.get("role", ""),
            "policy": manifest.get("policy", ""),
            "explicit_carrier_branch": manifest.get("explicit_carrier_branch", ""),
            "admission_mode": manifest.get("admission_mode", ""),
            "role_filter": manifest.get("role_filter", ""),
            "min_semantic_score": manifest.get("min_semantic_score", ""),
            "min_entry_age": manifest.get("min_entry_age", ""),
            "retention_budget": manifest.get("retention_budget", ""),
            "semantic_control": manifest.get("semantic_control", ""),
            "local_lane_mode": manifest.get("local_lane_mode", ""),
            "method": method,
            "gpu": manifest.get("gpu", ""),
            "worker_log": rel(worker_log),
            "worker_log_manifest_path": rel(Path(manifest["log"])),
            "worker_log_retry_used": worker_log_retry_used,
            "evaluate_log": rel(eval_log),
            "action_file": rel(Path(manifest["action_file"])),
            "fi_trace": rel(Path(manifest["fi_trace"])),
            "complete_exists": int((method_root / ".complete.json").exists()),
            "traj_exists": int((method_root / "traj.txt").exists()),
            "intrinsics_exists": int((method_root / "intrinsics.txt").exists()),
            "worker_log_success_markers": log_has_markers(worker_log, WORKER_SUCCESS_MARKERS),
            "evaluate_log_success_markers": log_has_markers(eval_log, EVALUATE_SUCCESS_MARKERS),
        }
        row.update(metrics.get(method, {}))
        row["metrics_present"] = all(finite_float(row.get(key)) for key in ("ate", "rpe_rot", "rpe_trans"))
        row.update(action_summary(Path(manifest["action_file"])))
        row.update(fi_summary(Path(manifest["fi_trace"])))
        row["runtime_row_pass"] = bool(
            row["metrics_present"]
            and row["worker_log_success_markers"]
            and row["evaluate_log_success_markers"]
            and row["raw_action_exact_frame_coverage_pass"]
            and row["complete_exists"]
            and row["traj_exists"]
            and row["intrinsics_exists"]
        )
        rows.append(row)
    return rows


def main() -> int:
    rows = build_rows()
    by_variant = {row["variant"]: row for row in rows}
    default = by_variant.get("cp0_default_no_policy", {})
    page = by_variant.get("cp1_page_qk_control", {})
    internal = by_variant.get("cp2_internal_qk_topk2", {}) or by_variant.get("cp2_internal_qk_topk4", {})
    shuffle = by_variant.get("cp7_provenance_shuffle_control", {})
    role_swap = by_variant.get("cp8_role_swap_control", {})
    reverse = by_variant.get("cp9_reverse_control", {})
    random = by_variant.get("cp10_random_control", {})
    comparison_rows = [default, page, internal, shuffle, role_swap, reverse, random]
    for row in rows:
        row["ate_improvement_pct_vs_default"] = pct_delta(row.get("ate"), default.get("ate"))
        row["ate_improvement_pct_vs_page"] = pct_delta(row.get("ate"), page.get("ate"))
        row["ate_improvement_pct_vs_internal"] = pct_delta(row.get("ate"), internal.get("ate"))
        row["ate_improvement_pct_vs_shuffle"] = pct_delta(row.get("ate"), shuffle.get("ate"))
        row["ate_improvement_pct_vs_reverse"] = pct_delta(row.get("ate"), reverse.get("ate"))
        row["ate_improvement_pct_vs_random"] = pct_delta(row.get("ate"), random.get("ate"))

    candidate_rows = [row for row in rows if row.get("role", "").startswith("form_")]
    control_rows = [row for row in rows if not row.get("role", "").startswith("form_")]
    active_candidates = [row for row in candidate_rows if row.get("runtime_explicit_evidence_pass")]
    best_candidate = min(
        [row for row in candidate_rows if finite_float(row.get("ate"))],
        key=lambda row: float(row["ate"]),
        default={},
    )
    def beats(row: dict[str, Any], other: dict[str, Any]) -> bool:
        return bool(finite_float(row.get("ate")) and finite_float(other.get("ate")) and float(row["ate"]) < float(other["ate"]))

    best_beats_controls = all(beats(best_candidate, control) for control in control_rows if control)
    finite_controls = [row for row in control_rows if finite_float(row.get("ate"))]
    best_control = min(finite_controls, key=lambda row: float(row["ate"]), default={})
    blocking_controls = [
        row
        for row in finite_controls
        if finite_float(best_candidate.get("ate")) and float(row["ate"]) <= float(best_candidate["ate"])
    ]
    best_blocking_control = min(blocking_controls, key=lambda row: float(row["ate"]), default={})
    best_geometry_pass = bool(
        finite_float(best_candidate.get("ate_improvement_pct_vs_default"))
        and float(best_candidate["ate_improvement_pct_vs_default"]) >= 1.0
    )
    form_count = len(candidate_rows)
    control_count = len(control_rows)
    terminal_pass = bool(
        form_count >= REQUIRED_FORM_COUNT
        and control_count >= REQUIRED_CONTROL_COUNT
        and all(bool(row.get("runtime_row_pass")) for row in rows)
        and active_candidates
        and best_geometry_pass
        and best_beats_controls
    )
    metric_scale_support_gate = ""
    if EXPLICIT_BRANCH == "C-LB-M":
        metric_scale_support_gate = "not_measured_in_seq00_minimech_no_anchor_calibrated_se3_or_scale_jump_metric"
        terminal_pass = False
    summary = {
        "schema": f"acl2_v119tf_stage3_{MINIMECH_LABEL}_minimech_summary_v1",
        "minimech_label": MINIMECH_LABEL,
        "explicit_carrier_branch": EXPLICIT_BRANCH,
        "run_root": rel(RUN_ROOT),
        "seq": SEQ,
        "dataset": DATASET,
        "dataset_tag": DATASET_TAG,
        "variant_count": len(rows),
        "form_count": form_count,
        "control_count": control_count,
        "required_form_count": REQUIRED_FORM_COUNT,
        "required_control_count": REQUIRED_CONTROL_COUNT,
        "runtime_worker_eval_all_pass": all(bool(row.get("runtime_row_pass")) for row in rows),
        "active_candidate_count": len(active_candidates),
        "best_candidate_variant": best_candidate.get("variant", ""),
        "best_candidate_ate": best_candidate.get("ate", ""),
        "default_ate": default.get("ate", ""),
        "page_ate": page.get("ate", ""),
        "internal_ate": internal.get("ate", ""),
        "provenance_shuffle_ate": shuffle.get("ate", ""),
        "role_swap_ate": role_swap.get("ate", ""),
        "reverse_ate": reverse.get("ate", ""),
        "random_ate": random.get("ate", ""),
        "road_ground_metric_qualification_control_ate": by_variant.get(
            "cp7_road_ground_metric_qualification_control", {}
        ).get("ate", ""),
        "low_baseline_repeated_view_control_ate": by_variant.get(
            "cp8_low_baseline_repeated_view_control", {}
        ).get("ate", ""),
        "dynamic_local_aligned_control_ate": by_variant.get(
            "cp9_dynamic_local_aligned_control", {}
        ).get("ate", ""),
        "random_metric_control_ate": by_variant.get("cp10_random_metric_control", {}).get("ate", ""),
        "best_control_variant": best_control.get("variant", ""),
        "best_control_ate": best_control.get("ate", ""),
        "best_blocking_control_variant": best_blocking_control.get("variant", ""),
        "best_blocking_control_ate": best_blocking_control.get("ate", ""),
        "candidate_ates_by_variant": json.dumps(
            {
                row["variant"]: row.get("ate", "")
                for row in candidate_rows
                if row.get("variant") and finite_float(row.get("ate"))
            },
            sort_keys=True,
        ),
        "control_ates_by_variant": json.dumps(
            {
                row["variant"]: row.get("ate", "")
                for row in control_rows
                if row.get("variant") and finite_float(row.get("ate"))
            },
            sort_keys=True,
        ),
        "best_candidate_geometry_pass_vs_default_1pct": best_geometry_pass,
        "best_candidate_beats_all_primary_controls": best_beats_controls,
        "metric_scale_support_gate": metric_scale_support_gate,
        "explicit_minimech_terminal_pass": terminal_pass,
        f"{MINIMECH_LABEL}_minimech_terminal_pass": terminal_pass,
        "truthfulness_boundary": (
            f"{EXPLICIT_BRANCH} seq00 minimum-mechanism matrix only; this does not satisfy full v119 completion, "
            "cross-sequence validation, dense geometry gates, or all Track C/HS branches."
        ),
    }
    summary_csv = RUN_ROOT / f"{SUMMARY_STEM}.csv"
    summary_json = RUN_ROOT / f"{SUMMARY_STEM}.json"
    root_csv = RESULT_ROOT / ROOT_EVIDENCE_NAME
    write_csv(summary_csv, rows)
    write_csv(root_csv, rows)
    summary.update({"summary_csv": rel(summary_csv), "summary_json": rel(summary_json), "root_evidence_csv": rel(root_csv)})
    summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
