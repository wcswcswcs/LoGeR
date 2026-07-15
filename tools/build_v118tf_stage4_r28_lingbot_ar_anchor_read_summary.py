#!/usr/bin/env python3
"""Summarize ACL2 v118 Stage4-R28 LingBot anchor-read pilot."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from statistics import median
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
_RESULT_ROOT_ENV = os.environ.get("ACL2_V118_AR_RESULT_ROOT", "").strip()
if _RESULT_ROOT_ENV:
    _result_root_path = Path(_RESULT_ROOT_ENV).expanduser()
    RESULT_ROOT = _result_root_path if _result_root_path.is_absolute() else ROOT / _result_root_path
else:
    RESULT_ROOT = ROOT / "results/acl2_v118tf_operation_specific_semantic_carrier_calibration"
STAGE_TAG = os.environ.get("ACL2_V118_AR_STAGE_TAG", "r28").strip().lower() or "r28"
DEFAULT_STAGE_SLUG = "stage4_r28_lingbot_ar_anchor_read"
STAGE_SLUG = os.environ.get("ACL2_V118_AR_STAGE_SLUG", DEFAULT_STAGE_SLUG).strip() or DEFAULT_STAGE_SLUG
STAGE = RESULT_ROOT / STAGE_SLUG
WORKSPACE = ROOT / "results/acl2_v105tf_dualtrack_lingbot_loger_evidence_eligibility/stage1_lingbot_baseline/workspace"
DATASET = "kitti_v105_00_01_02_05"
BASELINE_METHOD = "lingbot_map_stream_flashinfer_v118_r15_full"
MANIFEST = STAGE / f"summary/stage4_{STAGE_TAG}_lingbot_ar_anchor_read_manifest.json"
REGISTRY = RESULT_ROOT / "V118_RUN_REGISTRY.csv"
SEQS = tuple(
    part.strip().zfill(2)
    for part in os.environ.get("ACL2_V118_AR_SEQS", "00,02").replace(";", ",").split(",")
    if part.strip()
) or ("00", "02")


def rel(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def fnum(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def metric_path(seq: str, method: str) -> Path:
    return WORKSPACE / DATASET / seq / method / "eval/traj.json"


def complete_path(seq: str, method: str) -> Path:
    return WORKSPACE / DATASET / seq / method / ".complete.json"


def parse_frames(raw: Any) -> set[int]:
    out: set[int] = set()
    for part in str(raw or "").replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        try:
            out.add(int(part))
        except ValueError:
            pass
    return out


def action_trace_stats(
    path: Path,
    expected_frames: set[int],
    expected_query_roles: str,
    action_mode: str,
    expected_context_roles: set[str],
) -> dict[str, Any]:
    expected_row_type = (
        "anchor_source_value_scaling"
        if action_mode == "anchor_source_value_scaling"
        else "anchor_source_attention_weight"
    )
    rows = 0
    target_rows = 0
    changed_rows = 0
    applied_rows = 0
    observed_frames: set[int] = set()
    observed_query_roles: set[str] = set()
    observed_context_roles: set[str] = set()
    target_key_counts: list[int] = []
    changed_key_counts: list[int] = []
    target_query_counts: list[int] = []
    changed_query_key_counts: list[int] = []
    weight_mins: list[float] = []
    weight_maxs: list[float] = []
    weight_means: list[float] = []
    weight_stds: list[float] = []
    weight_l1s: list[float] = []
    weight_l2s: list[float] = []
    weight_delta_l1s: list[float] = []
    weight_delta_l2s: list[float] = []
    token_weight_key_counts: list[int] = []
    token_weight_rows = 0
    action_granularities: set[str] = set()
    value_weight_normalizations: set[str] = set()
    if not path.exists():
        return {
            "action_trace_exists": False,
            "action_log_rows": 0,
            "target_rows": 0,
            "changed_rows": 0,
            "attention_mask_applied_rows": 0,
            "value_scaling_applied_rows": 0,
            "observed_source_frames": "",
            "source_frame_coverage": 0.0 if expected_frames else "",
            "observed_query_roles": "",
            "expected_query_roles_seen": False,
            "observed_context_roles": "",
            "scale_reference_context_seen": False,
            "expected_context_roles_seen": False,
            "action_fidelity_pass": False,
        }
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("row_type") != expected_row_type:
                continue
            rows += 1
            target_key = int(row.get("target_key_count", row.get("target_value_token_count", 0)) or 0)
            changed_key = int(row.get("changed_key_count", row.get("changed_value_token_count", 0)) or 0)
            target_query = int(row.get("target_query_count", 0) or 0)
            changed_query_key = int(row.get("changed_query_key_count", row.get("changed_value_count", 0)) or 0)
            if target_key > 0:
                target_rows += 1
                target_key_counts.append(target_key)
                target_query_counts.append(target_query)
                observed_frames |= parse_frames(row.get("source_frames"))
                if row.get("query_roles"):
                    observed_query_roles.add(str(row.get("query_roles")))
                if row.get("source_context_role"):
                    observed_context_roles.add(str(row.get("source_context_role")))
                w_min = fnum(row.get("weight_min"))
                w_max = fnum(row.get("weight_max"))
                if w_min is not None:
                    weight_mins.append(w_min)
                if w_max is not None:
                    weight_maxs.append(w_max)
                for raw_key, bucket in [
                    ("weight_mean", weight_means),
                    ("weight_std", weight_stds),
                    ("weight_l1", weight_l1s),
                    ("weight_l2", weight_l2s),
                    ("weight_delta_l1", weight_delta_l1s),
                    ("weight_delta_l2", weight_delta_l2s),
                ]:
                    value = fnum(row.get(raw_key))
                    if value is not None:
                        bucket.append(value)
                if row.get("action_granularity"):
                    action_granularities.add(str(row.get("action_granularity")))
                if row.get("value_weight_normalization"):
                    value_weight_normalizations.add(str(row.get("value_weight_normalization")))
                try:
                    token_count = int(row.get("token_weight_key_count", 0) or 0)
                except (TypeError, ValueError):
                    token_count = 0
                token_weight_key_counts.append(token_count)
                if token_count > 0:
                    token_weight_rows += 1
            if changed_key > 0 and target_query > 0:
                changed_rows += 1
                changed_key_counts.append(changed_key)
                changed_query_key_counts.append(changed_query_key)
            if row.get("attention_mask_applied") is True or row.get("value_scaling_applied") is True:
                applied_rows += 1
    coverage = len(observed_frames & expected_frames) / len(expected_frames) if expected_frames else ""
    expected_query_seen = (
        True
        if action_mode == "anchor_source_value_scaling"
        else expected_query_roles in observed_query_roles
    )
    scale_context_seen = "scale_reference_context" in observed_context_roles
    expected_context_seen = bool(expected_context_roles) and expected_context_roles.issubset(observed_context_roles)
    action_fidelity_pass = bool(
        rows
        and target_rows
        and changed_rows
        and applied_rows
        and coverage == 1.0
        and expected_query_seen
        and expected_context_seen
    )
    return {
        "action_trace_exists": True,
        "action_log_rows": rows,
        "target_rows": target_rows,
        "changed_rows": changed_rows,
        "attention_mask_applied_rows": applied_rows if action_mode != "anchor_source_value_scaling" else 0,
        "value_scaling_applied_rows": applied_rows if action_mode == "anchor_source_value_scaling" else 0,
        "observed_source_frames": ";".join(str(frame) for frame in sorted(observed_frames)),
        "source_frame_coverage": coverage,
        "observed_query_roles": ";".join(sorted(observed_query_roles)),
        "expected_query_roles_seen": expected_query_seen,
        "observed_context_roles": ";".join(sorted(observed_context_roles)),
        "scale_reference_context_seen": scale_context_seen,
        "expected_context_roles": ";".join(sorted(expected_context_roles)),
        "expected_context_roles_seen": expected_context_seen,
        "target_key_count_min": min(target_key_counts) if target_key_counts else "",
        "target_key_count_max": max(target_key_counts) if target_key_counts else "",
        "changed_key_count_min": min(changed_key_counts) if changed_key_counts else "",
        "changed_key_count_max": max(changed_key_counts) if changed_key_counts else "",
        "target_query_count_min": min(target_query_counts) if target_query_counts else "",
        "target_query_count_max": max(target_query_counts) if target_query_counts else "",
        "changed_query_key_count_min": min(changed_query_key_counts) if changed_query_key_counts else "",
        "changed_query_key_count_max": max(changed_query_key_counts) if changed_query_key_counts else "",
        "weight_min_observed": min(weight_mins) if weight_mins else "",
        "weight_max_observed": max(weight_maxs) if weight_maxs else "",
        "weight_mean_min_observed": min(weight_means) if weight_means else "",
        "weight_mean_max_observed": max(weight_means) if weight_means else "",
        "weight_mean_median_observed": median(weight_means) if weight_means else "",
        "weight_std_min_observed": min(weight_stds) if weight_stds else "",
        "weight_std_max_observed": max(weight_stds) if weight_stds else "",
        "weight_std_median_observed": median(weight_stds) if weight_stds else "",
        "weight_l1_sum_observed": sum(weight_l1s) if weight_l1s else "",
        "weight_l2_max_observed": max(weight_l2s) if weight_l2s else "",
        "weight_delta_l1_sum_observed": sum(weight_delta_l1s) if weight_delta_l1s else "",
        "weight_delta_l2_max_observed": max(weight_delta_l2s) if weight_delta_l2s else "",
        "value_weight_normalization_observed": ";".join(sorted(value_weight_normalizations)),
        "action_granularity_observed": ";".join(sorted(action_granularities)),
        "token_weight_rows": token_weight_rows,
        "token_weight_key_count_min": min(token_weight_key_counts) if token_weight_key_counts else "",
        "token_weight_key_count_max": max(token_weight_key_counts) if token_weight_key_counts else "",
        "action_fidelity_pass": action_fidelity_pass,
    }


def summarize_rows(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    action_dir = STAGE / "runtime_full_thread8/action_traces"
    expected_frames = {int(frame) for frame in manifest.get("fixed_anchor_source_frames", [])}
    action_mode = str(manifest.get("action_mode", "anchor_source_attention_weight") or "anchor_source_attention_weight")
    expected_context_roles = {
        str(role)
        for role in manifest.get("source_context_roles", ["scale_reference_context"])
        if str(role)
    } or {"scale_reference_context"}
    branch = str(manifest.get("branch", "LB-AR") or "LB-AR")
    source_context_roles_text = ",".join(sorted(expected_context_roles))
    methods = manifest.get("methods", {})
    for method_base, meta in methods.items():
        for seq in SEQS:
            method = f"{method_base}_seq{seq}"
            eval_json = metric_path(seq, method)
            metrics = read_json(eval_json)
            baseline = read_json(metric_path(seq, BASELINE_METHOD))
            ate = fnum(metrics.get("ate"))
            baseline_ate = fnum(baseline.get("ate"))
            query_roles = ",".join(meta.get("query_roles", [])) if meta.get("query_roles") else "all"
            action_path = action_dir / f"{method}_seq{seq}.jsonl"
            rows.append(
                {
                    "schema": f"acl2_v118tf_stage4_{STAGE_TAG}_lingbot_ar_anchor_read_row_v1",
                    "dataset": DATASET,
                    "seq": seq,
                    "method_base": method_base,
                    "method": method,
                    "branch": branch,
                    "policy": meta.get("policy", ""),
                    "role": meta.get("role", ""),
                    "source_context_roles": source_context_roles_text,
                    "token_roles": "patch",
                    "query_roles": query_roles,
                    "source_frames": ";".join(str(frame) for frame in sorted(expected_frames)),
                    "eval_exists": eval_json.exists(),
                    "complete_marker_exists": complete_path(seq, method).exists(),
                    "ate": metrics.get("ate", ""),
                    "rpe_rot": metrics.get("rpe_rot", ""),
                    "rpe_trans": metrics.get("rpe_trans", ""),
                    "baseline_method": BASELINE_METHOD,
                    "baseline_ate": baseline.get("ate", ""),
                    "baseline_rpe_rot": baseline.get("rpe_rot", ""),
                    "baseline_rpe_trans": baseline.get("rpe_trans", ""),
                    "ate_rel_improvement_vs_default": (
                        (baseline_ate - ate) / baseline_ate
                        if ate is not None and baseline_ate not in (None, 0.0)
                        else ""
                    ),
                    "eval_json": rel(eval_json) if eval_json.exists() else str(eval_json),
                    "action_trace": rel(action_path) if action_path.exists() else str(action_path),
                    **action_trace_stats(action_path, expected_frames, query_roles, action_mode, expected_context_roles),
                }
            )
    return rows


def compare(rows: list[dict[str, Any]]) -> dict[str, Any]:
    candidate = {row["seq"]: row for row in rows if row.get("role") == "candidate"}
    controls = {
        role: {row["seq"]: row for row in rows if row.get("role") == role}
        for role in sorted({str(row.get("role")) for row in rows if row.get("role") not in {"candidate", ""}})
    }
    comparisons: dict[str, Any] = {}
    for role, control_rows in controls.items():
        common = sorted(candidate.keys() & control_rows.keys())
        metric_summary: dict[str, Any] = {}
        for metric in ("ate", "rpe_rot", "rpe_trans"):
            deltas = []
            per_seq = {}
            for seq in common:
                c = fnum(candidate[seq].get(metric))
                ctrl = fnum(control_rows[seq].get(metric))
                if c is None or ctrl is None:
                    continue
                delta = c - ctrl
                deltas.append(delta)
                per_seq[seq] = {"candidate": c, role: ctrl, "candidate_minus_control": delta}
            metric_summary[metric] = {
                "all_candidate_better_than_control": bool(deltas) and all(delta < 0 for delta in deltas),
                "median_candidate_minus_control": median(deltas) if deltas else "",
                "per_seq": per_seq,
            }
        comparisons[role] = {"seqs": common, "metrics": metric_summary}
    rels = [fnum(row.get("ate_rel_improvement_vs_default")) for row in candidate.values()]
    rels = [value for value in rels if value is not None]
    max_harm = abs(min(rels)) if rels and min(rels) < 0 else 0.0
    comparisons["baseline"] = {
        "ate_rel_improvement_vs_default": {
            "median": median(rels) if rels else "",
            "max_harm": max_harm,
            "both_sequences_improve": bool(rels) and len(rels) == len(SEQS) and all(value > 0 for value in rels),
            "pilot_gate": bool(rels) and len(rels) == len(SEQS) and median(rels) >= 0.03 and max_harm <= 0.01,
            "per_seq": {
                seq: fnum(row.get("ate_rel_improvement_vs_default"))
                for seq, row in sorted(candidate.items())
            },
        }
    }
    return comparisons


def row_action_fidelity_ok(row: dict[str, Any]) -> bool:
    if row.get("action_fidelity_pass") is True:
        return True
    policy = str(row.get("policy", "") or "").upper()
    role = str(row.get("role", "") or "").lower()
    if "NOOP" not in policy and "noop" not in role:
        return False
    try:
        target_rows = int(row.get("target_rows", 0) or 0)
    except (TypeError, ValueError):
        target_rows = 0
    try:
        changed_rows = int(row.get("changed_rows", 0) or 0)
    except (TypeError, ValueError):
        changed_rows = -1
    return bool(row.get("action_trace_exists") and target_rows > 0 and changed_rows == 0)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def add_registry_row(row: dict[str, Any]) -> None:
    rows = read_csv(REGISTRY)
    fields: list[str] = []
    for old in rows:
        for key in old:
            if key not in fields:
                fields.append(key)
    for key in row:
        if key not in fields:
            fields.append(key)
    kept = [
        old
        for old in rows
        if not (
            old.get("stage") == row.get("stage")
            and old.get("surface_or_branch") == row.get("surface_or_branch")
            and old.get("artifact") == row.get("artifact")
        )
    ]
    kept.append({key: row.get(key, "") for key in fields})
    with REGISTRY.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(kept)


def main() -> None:
    manifest = read_json(MANIFEST)
    if not manifest:
        raise FileNotFoundError(MANIFEST)
    action_mode = str(manifest.get("action_mode", "anchor_source_attention_weight") or "anchor_source_attention_weight")
    intervention_form = str(manifest.get("intervention_form", "selected_query_attention_weight") or "")
    requires_token_weight_key_count = bool(manifest.get("requires_token_weight_key_count", False))
    rows = summarize_rows(manifest)
    complete = all(row["eval_exists"] and row["complete_marker_exists"] and row["action_trace_exists"] for row in rows)
    token_weight_fidelity = (
        True
        if not requires_token_weight_key_count
        else bool(rows)
        and all(
            int(row.get("token_weight_rows", 0) or 0) > 0
            and int(row.get("token_weight_key_count_min", 0) or 0) > 0
            for row in rows
        )
    )
    action_fidelity = all(row_action_fidelity_ok(row) for row in rows) and token_weight_fidelity
    comparisons = compare(rows)
    control_roles = sorted(role for role in comparisons if role != "baseline")
    candidate_better_all_controls = all(
        comparisons[role]["metrics"]["ate"]["all_candidate_better_than_control"]
        for role in control_roles
    )
    candidate_better_all_controls = bool(control_roles) and candidate_better_all_controls
    baseline_gate = comparisons["baseline"]["ate_rel_improvement_vs_default"]["pilot_gate"]
    branch = str(manifest.get("branch", "LB-AR") or "LB-AR")
    if branch == "LB-LR":
        decision_prefix = "LR_SOURCE_VALUE_ROUTING" if action_mode == "anchor_source_value_scaling" else "LR_LOCAL_READ"
    else:
        decision_prefix = "AR_SOURCE_VALUE_SCALING" if action_mode == "anchor_source_value_scaling" else "AR_ANCHOR_READ"
    if not complete:
        decision = f"{decision_prefix}_RUNTIME_INCOMPLETE"
    elif not action_fidelity:
        decision = f"{decision_prefix}_ACTION_FIDELITY_NO_GO"
    elif candidate_better_all_controls and baseline_gate:
        decision = f"{decision_prefix}_CANDIDATE_FOUND_REQUIRES_FRESH_VALIDATION"
    else:
        decision = f"{decision_prefix}_CONTROL_OR_BASELINE_NO_GO"

    out = STAGE / "summary"
    out.mkdir(parents=True, exist_ok=True)
    rows_path = out / f"stage4_{STAGE_TAG}_lingbot_ar_anchor_read_rows.csv"
    summary_path = out / f"stage4_{STAGE_TAG}_lingbot_ar_anchor_read_summary.json"
    report_path = out / f"STAGE4_{STAGE_TAG.upper()}_LINGBOT_AR_ANCHOR_READ_REPORT.md"
    write_csv(rows_path, rows)
    summary = {
        "schema": f"acl2_v118tf_stage4_{STAGE_TAG}_lingbot_ar_anchor_read_summary_v1",
        f"stage4_{STAGE_TAG}_decision": decision,
        "global_goal_achieved": False,
        "dataset": DATASET,
        "baseline_method": BASELINE_METHOD,
        "branch": branch,
        "operation": manifest.get("operation", "Anchor read"),
        "source_context_roles": manifest.get("source_context_roles", ["scale_reference_context"]),
        "action_mode": action_mode,
        "intervention_form": intervention_form,
        "value_weight_normalization": manifest.get("value_weight_normalization", ""),
        "complete": complete,
        "action_fidelity": action_fidelity,
        "requires_token_weight_key_count": requires_token_weight_key_count,
        "token_weight_fidelity": token_weight_fidelity,
        "row_count": len(rows),
        "candidate_better_all_controls": candidate_better_all_controls,
        "baseline_gate": baseline_gate,
        "comparisons": comparisons,
        "boundary": manifest.get(
            "boundary",
            "LB-AR 00/02 dev pilot; a pass is not a global success until fresh validation and required AR variants complete.",
        ),
        "outputs": {
            "rows": rel(rows_path),
            "summary": rel(summary_path),
            "report": rel(report_path),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    report = [
        f"# Stage4-{STAGE_TAG.upper()} LingBot {branch} {manifest.get('operation', 'Anchor read')} Summary",
        "",
        f"decision: `{decision}`",
        f"global_goal_achieved: `{summary['global_goal_achieved']}`",
        f"complete: `{complete}`",
        f"action_fidelity: `{action_fidelity}`",
        "",
        "```json",
        json.dumps({"comparisons": comparisons}, indent=2, sort_keys=True),
        "```",
    ]
    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    add_registry_row(
        {
            "schema": "acl2_v118tf_run_registry_row_v1",
            "stage": f"Stage4-{STAGE_TAG.upper()}",
            "surface_or_branch": branch,
            "status": decision,
            "artifact": rel(summary_path),
            "notes": f"{branch} {intervention_form or action_mode} pilot on fixed source frames; dev-only 00/02 selector, global success remains false",
        }
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
