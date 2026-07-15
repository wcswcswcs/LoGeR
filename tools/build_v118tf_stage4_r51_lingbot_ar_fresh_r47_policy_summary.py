#!/usr/bin/env python3
"""Summarize ACL2 v118 Stage4-R51 fresh R47-policy validation."""

from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
from statistics import median
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "results/acl2_v118tf_operation_specific_semantic_carrier_calibration"


def parse_seq_env(name: str, default: str) -> tuple[str, ...]:
    seqs = tuple(part.strip().zfill(2) for part in os.environ.get(name, default).replace(";", ",").split(",") if part.strip())
    return seqs or tuple(part.strip().zfill(2) for part in default.split(",") if part.strip())


STAGE_TAG = os.environ.get("ACL2_V118_FRESH_POLICY_TAG", "r51").strip().lower() or "r51"
STAGE_LABEL = STAGE_TAG.upper()
STAGE = RESULT_ROOT / os.environ.get("ACL2_V118_FRESH_POLICY_STAGE_SLUG", "stage4_r51_lingbot_ar_fresh_r47_policy_validation")
SUMMARY_DIR = STAGE / "summary"
R49 = RESULT_ROOT / os.environ.get("ACL2_V118_FRESH_POLICY_TRACE_STAGE_SLUG", "stage4_r49_lingbot_ar_fresh_trace_baseline")
WORKSPACE = R49 / "workspace"
BASELINE_METHOD = os.environ.get("ACL2_V118_FRESH_POLICY_BASELINE_METHOD", "lingbot_map_stream_flashinfer_v118_r49_fresh_trace")
SEQS = parse_seq_env("ACL2_V118_FRESH_POLICY_SEQS", "04,03")
SEQ_LABEL = "/".join(SEQS)
DATASET_PREFIX = os.environ.get("ACL2_V118_FRESH_DATASET_PREFIX", "kitti_v118_r49_fresh_seq")
POLICY_RULE = os.environ.get("ACL2_V118_FRESH_POLICY_RULE", "r47").strip().lower() or "r47"
RISK_MIN_CORR = float(os.environ.get("ACL2_V118_FRESH_POLICY_RISK_MIN_CORR", "0.50"))
RISK_MIN_STABLE_TO_WEAK_LOWTRUST = float(os.environ.get("ACL2_V118_FRESH_POLICY_RISK_MIN_STABLE_TO_WEAK_LOWTRUST", "0.20"))
ACTION_DYNAMIC_MIN = float(os.environ.get("ACL2_V118_FRESH_POLICY_ACTION_DYNAMIC_MIN", "0.24"))
NEGATIVE_CORR_RISK_MIN_RATIO = float(os.environ.get("ACL2_V118_FRESH_POLICY_NEGATIVE_CORR_RISK_MIN_RATIO", "0.08"))
NEGATIVE_CORR_RISK_MIN_DYNAMIC = float(os.environ.get("ACL2_V118_FRESH_POLICY_NEGATIVE_CORR_RISK_MIN_DYNAMIC", "0.18"))
STABLE_BOOST_MIN_RATIO = float(os.environ.get("ACL2_V118_FRESH_POLICY_STABLE_BOOST_MIN_RATIO", "0.20"))
STABLE_BOOST_MIN_STABLE_MEAN = float(os.environ.get("ACL2_V118_FRESH_POLICY_STABLE_BOOST_MIN_STABLE_MEAN", "0.15"))
CONTROL_SAFE_NEGATIVE_RATIO_LOW = float(os.environ.get("ACL2_V118_FRESH_POLICY_CONTROL_SAFE_NEGATIVE_RATIO_LOW", "0.05"))
CONTROL_SAFE_NEGATIVE_RATIO_HIGH = float(os.environ.get("ACL2_V118_FRESH_POLICY_CONTROL_SAFE_NEGATIVE_RATIO_HIGH", "0.20"))
CONTROL_SAFE_RISK_MIN_CORR = float(os.environ.get("ACL2_V118_FRESH_POLICY_CONTROL_SAFE_RISK_MIN_CORR", "0.75"))
CONTROL_SAFE_RISK_MIN_RATIO = float(os.environ.get("ACL2_V118_FRESH_POLICY_CONTROL_SAFE_RISK_MIN_RATIO", "0.20"))
CONTROL_SAFE_RISK_MIN_DYNAMIC = float(os.environ.get("ACL2_V118_FRESH_POLICY_CONTROL_SAFE_RISK_MIN_DYNAMIC", "0.20"))
CONTROL_SAFE_V2_STRONG_NEG_CORR = float(os.environ.get("ACL2_V118_FRESH_POLICY_CONTROL_SAFE_V2_STRONG_NEG_CORR", "-0.40"))
CONTROL_SAFE_V2_MID_RATIO_LOW = float(os.environ.get("ACL2_V118_FRESH_POLICY_CONTROL_SAFE_V2_MID_RATIO_LOW", "0.08"))
CONTROL_SAFE_V2_MID_RATIO_HIGH = float(os.environ.get("ACL2_V118_FRESH_POLICY_CONTROL_SAFE_V2_MID_RATIO_HIGH", "0.20"))
CONTROL_SAFE_V2_POSITIVE_LOW_RATIO_MAX = float(os.environ.get("ACL2_V118_FRESH_POLICY_CONTROL_SAFE_V2_POSITIVE_LOW_RATIO_MAX", "0.08"))
CONTROL_SAFE_V2_POSITIVE_LOW_DYNAMIC_MAX = float(os.environ.get("ACL2_V118_FRESH_POLICY_CONTROL_SAFE_V2_POSITIVE_LOW_DYNAMIC_MAX", "0.20"))
CONTROL_SAFE_V2_STABLE_BOOST_CORR_MIN = float(os.environ.get("ACL2_V118_FRESH_POLICY_CONTROL_SAFE_V2_STABLE_BOOST_CORR_MIN", "0.65"))
CONTROL_SAFE_V2_STABLE_BOOST_RATIO_MAX = float(os.environ.get("ACL2_V118_FRESH_POLICY_CONTROL_SAFE_V2_STABLE_BOOST_RATIO_MAX", "0.16"))
CONTROL_SAFE_V2_STABLE_BOOST_DYNAMIC_MIN = float(os.environ.get("ACL2_V118_FRESH_POLICY_CONTROL_SAFE_V2_STABLE_BOOST_DYNAMIC_MIN", "0.24"))
CONTROL_SAFE_V2_MODERATE_REVERSE_CORR_MAX = float(os.environ.get("ACL2_V118_FRESH_POLICY_CONTROL_SAFE_V2_MODERATE_REVERSE_CORR_MAX", "0.50"))
ANCHOR_FRAMES = set(range(8))
POLICY_ROWS = SUMMARY_DIR / "stage4_r51_fresh_policy_rows.csv"
METHOD_MANIFEST = SUMMARY_DIR / "stage4_r51_fresh_policy_method_manifest.csv"


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def clean_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clean_json(v) for v in value]
    if isinstance(value, Path):
        return rel(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def fnum(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def dataset_name(seq: str) -> str:
    return f"{DATASET_PREFIX}{seq}"


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def policy_rule_lines() -> list[str]:
    if POLICY_RULE == "control_safe_boundary_v3":
        return [
            f"if corr <= {CONTROL_SAFE_V2_STRONG_NEG_CORR:.2f} and {CONTROL_SAFE_V2_MID_RATIO_LOW:.2f} <= stable_to_weak_lowtrust < {CONTROL_SAFE_V2_MID_RATIO_HIGH:.2f}: risk",
            f"elif corr <= 0 and stable_to_weak_lowtrust < {CONTROL_SAFE_NEGATIVE_RATIO_LOW:.2f}: reverse",
            f"elif corr <= 0 and stable_to_weak_lowtrust >= {CONTROL_SAFE_NEGATIVE_RATIO_HIGH:.2f}: reverse",
            "elif corr <= 0: abstain",
            (
                f"elif corr >= {CONTROL_SAFE_RISK_MIN_CORR:.2f} "
                f"and stable_to_weak_lowtrust >= {CONTROL_SAFE_RISK_MIN_RATIO:.2f} "
                f"and dynamic_plus_lowtrust_mean >= {CONTROL_SAFE_RISK_MIN_DYNAMIC:.2f}: risk"
            ),
            f"elif corr >= 0.65 and stable_to_weak_lowtrust < {CONTROL_SAFE_V2_POSITIVE_LOW_RATIO_MAX:.2f} and dynamic_plus_lowtrust_mean < {CONTROL_SAFE_V2_POSITIVE_LOW_DYNAMIC_MAX:.2f}: risk",
            (
                f"elif corr >= {CONTROL_SAFE_V2_STABLE_BOOST_CORR_MIN:.2f} "
                f"and {CONTROL_SAFE_V2_MID_RATIO_LOW:.2f} <= stable_to_weak_lowtrust < {CONTROL_SAFE_V2_STABLE_BOOST_RATIO_MAX:.2f} "
                f"and dynamic_plus_lowtrust_mean >= {CONTROL_SAFE_V2_STABLE_BOOST_DYNAMIC_MIN:.2f}: risk_only"
            ),
            (
                f"elif corr < {CONTROL_SAFE_V2_MODERATE_REVERSE_CORR_MAX:.2f} "
                f"and stable_to_weak_lowtrust >= {CONTROL_SAFE_RISK_MIN_RATIO:.2f} "
                f"and dynamic_plus_lowtrust_mean >= {CONTROL_SAFE_RISK_MIN_DYNAMIC:.2f}: reverse"
            ),
            "else: abstain",
        ]
    if POLICY_RULE == "control_safe_boundary_v2":
        return [
            f"if corr <= {CONTROL_SAFE_V2_STRONG_NEG_CORR:.2f} and {CONTROL_SAFE_V2_MID_RATIO_LOW:.2f} <= stable_to_weak_lowtrust < {CONTROL_SAFE_V2_MID_RATIO_HIGH:.2f}: risk",
            f"elif corr <= 0 and stable_to_weak_lowtrust < {CONTROL_SAFE_NEGATIVE_RATIO_LOW:.2f}: reverse",
            f"elif corr <= 0 and stable_to_weak_lowtrust >= {CONTROL_SAFE_NEGATIVE_RATIO_HIGH:.2f}: reverse",
            "elif corr <= 0: abstain",
            (
                f"elif corr >= {CONTROL_SAFE_RISK_MIN_CORR:.2f} "
                f"and stable_to_weak_lowtrust >= {CONTROL_SAFE_RISK_MIN_RATIO:.2f} "
                f"and dynamic_plus_lowtrust_mean >= {CONTROL_SAFE_RISK_MIN_DYNAMIC:.2f}: risk"
            ),
            f"elif corr >= 0.65 and stable_to_weak_lowtrust < {CONTROL_SAFE_V2_POSITIVE_LOW_RATIO_MAX:.2f} and dynamic_plus_lowtrust_mean < {CONTROL_SAFE_V2_POSITIVE_LOW_DYNAMIC_MAX:.2f}: risk",
            (
                f"elif corr >= {CONTROL_SAFE_V2_STABLE_BOOST_CORR_MIN:.2f} "
                f"and {CONTROL_SAFE_V2_MID_RATIO_LOW:.2f} <= stable_to_weak_lowtrust < {CONTROL_SAFE_V2_STABLE_BOOST_RATIO_MAX:.2f} "
                f"and dynamic_plus_lowtrust_mean >= {CONTROL_SAFE_V2_STABLE_BOOST_DYNAMIC_MIN:.2f}: stable_boost"
            ),
            (
                f"elif corr < {CONTROL_SAFE_V2_MODERATE_REVERSE_CORR_MAX:.2f} "
                f"and stable_to_weak_lowtrust >= {CONTROL_SAFE_RISK_MIN_RATIO:.2f} "
                f"and dynamic_plus_lowtrust_mean >= {CONTROL_SAFE_RISK_MIN_DYNAMIC:.2f}: reverse"
            ),
            "else: abstain",
        ]
    if POLICY_RULE == "control_safe_boundary":
        return [
            f"if corr <= 0 and stable_to_weak_lowtrust < {CONTROL_SAFE_NEGATIVE_RATIO_LOW:.2f}: reverse",
            f"elif corr <= 0 and stable_to_weak_lowtrust >= {CONTROL_SAFE_NEGATIVE_RATIO_HIGH:.2f}: reverse",
            f"elif corr <= 0: abstain",
            (
                f"elif corr >= {CONTROL_SAFE_RISK_MIN_CORR:.2f} "
                f"and stable_to_weak_lowtrust >= {CONTROL_SAFE_RISK_MIN_RATIO:.2f} "
                f"and dynamic_plus_lowtrust_mean >= {CONTROL_SAFE_RISK_MIN_DYNAMIC:.2f}: risk"
            ),
            "else: abstain",
        ]
    if POLICY_RULE == "stable_dominant_boost":
        return [
            f"if stable_to_weak_lowtrust >= {STABLE_BOOST_MIN_RATIO:.2f} and stable_mean >= {STABLE_BOOST_MIN_STABLE_MEAN:.2f}: stable_boost",
            "elif corr <= 0: reverse",
            f"elif corr >= {RISK_MIN_CORR:.2f}: risk",
            "else: abstain",
        ]
    if POLICY_RULE == "regime_action_sensitivity":
        return [
            f"if corr <= 0 and (stable_to_weak_lowtrust >= {NEGATIVE_CORR_RISK_MIN_RATIO:.2f} or dynamic_plus_lowtrust_mean >= {NEGATIVE_CORR_RISK_MIN_DYNAMIC:.2f}): risk",
            "elif corr <= 0: reverse",
            f"elif corr >= {RISK_MIN_CORR:.2f} and stable_to_weak_lowtrust >= {RISK_MIN_STABLE_TO_WEAK_LOWTRUST:.2f}: risk",
            f"elif corr >= {RISK_MIN_CORR:.2f} and dynamic_plus_lowtrust_mean >= {ACTION_DYNAMIC_MIN:.2f}: risk",
            f"elif corr >= {RISK_MIN_CORR:.2f}: abstain",
            "elif stable_to_weak_lowtrust >= 0.20: reverse",
            "else: abstain",
        ]
    if POLICY_RULE == "stable_guarded_risk":
        return [
            "if corr <= 0: reverse",
            f"elif corr >= {RISK_MIN_CORR:.2f} and stable_to_weak_lowtrust >= {RISK_MIN_STABLE_TO_WEAK_LOWTRUST:.2f}: risk",
            f"elif corr >= {RISK_MIN_CORR:.2f}: abstain",
            "elif stable_to_weak_lowtrust >= 0.20: reverse",
            "else: abstain",
        ]
    return [
        "if corr <= 0: reverse",
        "elif corr >= 0.50: risk",
        "elif stable_to_weak_lowtrust >= 0.20: reverse",
        "else: abstain",
    ]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean_json(data), indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def metric_path(seq: str, method: str) -> Path:
    return WORKSPACE / dataset_name(seq) / seq / method / "eval/traj.json"


def complete_path(seq: str, method: str) -> Path:
    return WORKSPACE / dataset_name(seq) / seq / method / ".complete.json"


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


def action_trace_stats(path: Path) -> dict[str, Any]:
    rows = 0
    target_rows = 0
    changed_rows = 0
    applied_rows = 0
    observed_frames: set[int] = set()
    observed_context_roles: set[str] = set()
    observed_token_modes: set[str] = set()
    token_weight_key_counts: list[int] = []
    changed_value_token_counts: list[int] = []
    weight_mins: list[float] = []
    weight_maxs: list[float] = []
    if not path.exists():
        return {
            "action_trace_exists": False,
            "action_log_rows": 0,
            "target_rows": 0,
            "changed_rows": 0,
            "value_scaling_applied_rows": 0,
            "observed_source_frames": "",
            "source_frame_coverage": 0.0,
            "scale_reference_context_seen": False,
            "token_weight_rows": 0,
            "token_weight_key_count_min": "",
            "token_weight_key_count_max": "",
            "action_fidelity_pass": False,
        }
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("row_type") != "anchor_source_value_scaling":
                continue
            rows += 1
            target = int(row.get("target_value_token_count", 0) or 0)
            changed = int(row.get("changed_value_token_count", 0) or 0)
            token_count = int(row.get("token_weight_key_count", 0) or 0)
            if target > 0:
                target_rows += 1
                observed_frames |= parse_frames(row.get("source_frames"))
                token_weight_key_counts.append(token_count)
                if row.get("source_context_role"):
                    observed_context_roles.add(str(row.get("source_context_role")))
                if row.get("token_weight_mode"):
                    observed_token_modes.add(str(row.get("token_weight_mode")))
                w_min = fnum(row.get("weight_min"))
                w_max = fnum(row.get("weight_max"))
                if w_min is not None:
                    weight_mins.append(w_min)
                if w_max is not None:
                    weight_maxs.append(w_max)
            if changed > 0:
                changed_rows += 1
                changed_value_token_counts.append(changed)
            if row.get("value_scaling_applied") is True:
                applied_rows += 1
    coverage = len(observed_frames & ANCHOR_FRAMES) / len(ANCHOR_FRAMES)
    token_weight_rows = sum(1 for value in token_weight_key_counts if value > 0)
    action_fidelity_pass = bool(
        rows
        and target_rows
        and changed_rows
        and applied_rows
        and coverage == 1.0
        and "scale_reference_context" in observed_context_roles
        and token_weight_rows > 0
    )
    return {
        "action_trace_exists": True,
        "action_log_rows": rows,
        "target_rows": target_rows,
        "changed_rows": changed_rows,
        "value_scaling_applied_rows": applied_rows,
        "observed_source_frames": ";".join(str(frame) for frame in sorted(observed_frames)),
        "source_frame_coverage": coverage,
        "observed_context_roles": ";".join(sorted(observed_context_roles)),
        "scale_reference_context_seen": "scale_reference_context" in observed_context_roles,
        "observed_token_weight_modes": ";".join(sorted(observed_token_modes)),
        "changed_value_token_count_min": min(changed_value_token_counts) if changed_value_token_counts else "",
        "changed_value_token_count_max": max(changed_value_token_counts) if changed_value_token_counts else "",
        "weight_min_observed": min(weight_mins) if weight_mins else "",
        "weight_max_observed": max(weight_maxs) if weight_maxs else "",
        "token_weight_rows": token_weight_rows,
        "token_weight_key_count_min": min(token_weight_key_counts) if token_weight_key_counts else "",
        "token_weight_key_count_max": max(token_weight_key_counts) if token_weight_key_counts else "",
        "action_fidelity_pass": action_fidelity_pass,
    }


def summarize_runtime_rows(method_manifest: list[dict[str, str]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in method_manifest:
        seq = str(spec["seq"]).zfill(2)
        method = spec["method"]
        eval_json = metric_path(seq, method)
        metrics = read_json(eval_json)
        baseline_json = metric_path(seq, BASELINE_METHOD)
        baseline_metrics = read_json(baseline_json)
        ate = fnum(metrics.get("ate"))
        baseline_ate = fnum(baseline_metrics.get("ate"))
        action_path = ROOT / spec["action_trace"] if not Path(spec["action_trace"]).is_absolute() else Path(spec["action_trace"])
        rows.append(
            {
                "schema": "acl2_v118tf_stage4_r51_fresh_policy_runtime_row_v1",
                "seq": seq,
                "dataset": dataset_name(seq),
                "method": method,
                "role": spec.get("role", ""),
                "policy": spec.get("policy", ""),
                "policy_action": spec.get("action", ""),
                "token_weight_mode": spec.get("token_weight_mode", ""),
                "eval_exists": eval_json.is_file(),
                "complete_marker_exists": complete_path(seq, method).is_file(),
                "ate": metrics.get("ate", ""),
                "rpe_rot": metrics.get("rpe_rot", ""),
                "rpe_trans": metrics.get("rpe_trans", ""),
                "baseline_method": BASELINE_METHOD,
                "baseline_ate": baseline_metrics.get("ate", ""),
                "baseline_rpe_rot": baseline_metrics.get("rpe_rot", ""),
                "baseline_rpe_trans": baseline_metrics.get("rpe_trans", ""),
                "ate_rel_improvement_vs_default": (
                    (baseline_ate - ate) / baseline_ate
                    if ate is not None and baseline_ate not in (None, 0.0)
                    else ""
                ),
                "eval_json": rel(eval_json) if eval_json.exists() else str(eval_json),
                "action_trace": rel(action_path) if action_path.exists() else str(action_path),
                **action_trace_stats(action_path),
            }
        )
    return rows


def baseline_metric_row(seq: str) -> dict[str, Any]:
    metrics = read_json(metric_path(seq, BASELINE_METHOD))
    return {
        "seq": seq,
        "method": BASELINE_METHOD,
        "role": "baseline_selected_by_abstention",
        "ate": metrics.get("ate", ""),
        "rpe_rot": metrics.get("rpe_rot", ""),
        "rpe_trans": metrics.get("rpe_trans", ""),
        "baseline_ate": metrics.get("ate", ""),
        "ate_rel_improvement_vs_default": 0.0,
        "eval_exists": bool(metrics),
        "complete_marker_exists": complete_path(seq, BASELINE_METHOD).is_file(),
        "action_fidelity_pass": True,
    }


def compare(policy_rows: list[dict[str, str]], runtime_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_seq: dict[str, list[dict[str, Any]]] = {}
    for row in runtime_rows:
        by_seq.setdefault(str(row["seq"]).zfill(2), []).append(row)
    comparison_rows: list[dict[str, Any]] = []
    selected_rels: list[float] = []
    control_deltas: list[float] = []
    selected_improvement_count = 0
    abstain_count = 0
    for policy in policy_rows:
        seq = str(policy["seq"]).zfill(2)
        action = policy.get("policy_action", "")
        seq_rows = by_seq.get(seq, [])
        if action == "abstain":
            selected = baseline_metric_row(seq)
            abstain_count += 1
        else:
            selected = next((row for row in seq_rows if row.get("role") == "candidate"), {})
        selected_ate = fnum(selected.get("ate"))
        baseline_ate = fnum(selected.get("baseline_ate"))
        selected_rel = fnum(selected.get("ate_rel_improvement_vs_default"))
        if selected_rel is None and selected_ate is not None and baseline_ate not in (None, 0.0):
            selected_rel = (baseline_ate - selected_ate) / baseline_ate
        if selected_rel is not None:
            selected_rels.append(selected_rel)
            if selected_rel > 0:
                selected_improvement_count += 1
        controls = [row for row in seq_rows if row.get("role") != "candidate"]
        if not controls:
            comparison_rows.append(
                {
                    "schema": "acl2_v118tf_stage4_r51_fresh_policy_comparison_row_v1",
                    "seq": seq,
                    "policy_action": action,
                    "selected_method": selected.get("method", ""),
                    "selected_ate": selected.get("ate", ""),
                    "selected_rel_vs_default": selected_rel if selected_rel is not None else "",
                    "control_role": "",
                    "control_method": "",
                    "control_ate": "",
                    "selected_minus_control_ate": "",
                    "selected_better_control": "",
                }
            )
        for control in controls:
            control_ate = fnum(control.get("ate"))
            delta = None if selected_ate is None or control_ate is None else selected_ate - control_ate
            if delta is not None:
                control_deltas.append(delta)
            comparison_rows.append(
                {
                    "schema": "acl2_v118tf_stage4_r51_fresh_policy_comparison_row_v1",
                    "seq": seq,
                    "policy_action": action,
                    "selected_method": selected.get("method", ""),
                    "selected_ate": selected.get("ate", ""),
                    "selected_rel_vs_default": selected_rel if selected_rel is not None else "",
                    "control_role": control.get("role", ""),
                    "control_method": control.get("method", ""),
                    "control_ate": control.get("ate", ""),
                    "selected_minus_control_ate": delta if delta is not None else "",
                    "selected_better_control": (delta < 0.0) if delta is not None else "",
                }
            )
    all_nonharm = bool(selected_rels) and len(selected_rels) == len(policy_rows) and all(value >= -1e-12 for value in selected_rels)
    candidate_better_controls = bool(control_deltas) and all(delta < 0.0 for delta in control_deltas)
    return comparison_rows, {
        "selected_rels": selected_rels,
        "selected_improvement_count": selected_improvement_count,
        "abstain_count": abstain_count,
        "all_sequences_nonharm": all_nonharm,
        "candidate_better_all_controls": candidate_better_controls,
        "median_rel_vs_default": median(selected_rels) if selected_rels else "",
        "max_harm": abs(min(selected_rels)) if selected_rels and min(selected_rels) < 0 else 0.0,
        "median_selected_minus_control_ate": median(control_deltas) if control_deltas else "",
    }


def main() -> int:
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    policy_rows = read_csv(POLICY_ROWS)
    method_manifest = read_csv(METHOD_MANIFEST)
    if not policy_rows:
        raise FileNotFoundError(POLICY_ROWS)
    if not method_manifest:
        raise FileNotFoundError(METHOD_MANIFEST)
    runtime_rows = summarize_runtime_rows(method_manifest)
    comparison_rows, comparison_summary = compare(policy_rows, runtime_rows)
    complete = bool(runtime_rows) and all(row["eval_exists"] and row["complete_marker_exists"] for row in runtime_rows)
    action_fidelity = bool(runtime_rows) and all(row.get("action_fidelity_pass") is True for row in runtime_rows)
    median_rel = fnum(comparison_summary["median_rel_vs_default"])
    baseline_gate = bool(
        comparison_summary["all_sequences_nonharm"]
        and median_rel is not None
        and median_rel >= 0.03
        and fnum(comparison_summary["max_harm"]) == 0.0
    )
    policy_gate = bool(
        complete
        and action_fidelity
        and comparison_summary["candidate_better_all_controls"]
        and baseline_gate
    )
    if not complete:
        decision = "FRESH_R47_POLICY_RUNTIME_INCOMPLETE"
    elif not action_fidelity:
        decision = "FRESH_R47_POLICY_ACTION_FIDELITY_NO_GO"
    elif policy_gate:
        decision = "FRESH_R47_POLICY_VALIDATION_PASS_BRANCH_ONLY_GLOBAL_FALSE"
    else:
        decision = "FRESH_R47_POLICY_CONTROL_OR_BASELINE_NO_GO"

    rows_path = SUMMARY_DIR / "stage4_r51_fresh_policy_runtime_rows.csv"
    comparison_path = SUMMARY_DIR / "stage4_r51_fresh_policy_comparison_rows.csv"
    summary_path = SUMMARY_DIR / "stage4_r51_fresh_policy_summary.json"
    report_path = SUMMARY_DIR / "STAGE4_R51_FRESH_R47_POLICY_REPORT.md"
    write_csv(rows_path, runtime_rows)
    write_csv(comparison_path, comparison_rows)
    summary = {
        "schema": "acl2_v118tf_stage4_r51_fresh_r47_policy_summary_v1",
        f"stage4_{STAGE_TAG}_decision": decision,
        "global_goal_achieved": False,
        "claim_level": f"fresh_{'_'.join(SEQS)}_branch_validation_only_not_global_v118_success",
        "complete": complete,
        "action_fidelity": action_fidelity,
        "policy_gate": policy_gate,
        "baseline_gate": baseline_gate,
        "candidate_better_all_controls": comparison_summary["candidate_better_all_controls"],
        "all_sequences_nonharm": comparison_summary["all_sequences_nonharm"],
        "selected_improvement_count": comparison_summary["selected_improvement_count"],
        "abstain_count": comparison_summary["abstain_count"],
        "sequence_count": len(policy_rows),
        "method_count": len(runtime_rows),
        "median_rel_vs_default": comparison_summary["median_rel_vs_default"],
        "max_harm": comparison_summary["max_harm"],
        "median_selected_minus_control_ate": comparison_summary["median_selected_minus_control_ate"],
        "policy_rows": policy_rows,
        "comparison_rows": comparison_rows,
        "rule": {
            "name": POLICY_RULE,
            "inputs": ["internal_semantic_corr", "stable_to_weak_lowtrust", "dynamic_plus_lowtrust_mean"],
            "logic": policy_rule_lines(),
            "validation_boundary": (
                f"fresh KITTI {SEQ_LABEL}, selected after R48 readiness scan from previously unused v118 fresh candidates"
            ),
        },
        "outputs": {
            "runtime_rows": rel(rows_path),
            "comparison_rows": rel(comparison_path),
            "summary": rel(summary_path),
            "report": rel(report_path),
        },
        "boundary": (
            f"{STAGE_LABEL} validates the R47 rule on fresh {SEQ_LABEL} only. Even a pass is not the global v118 success because the "
            "full plan still requires broader branch/surface coverage and promoted-candidate controls."
        ),
    }
    write_json(summary_path, summary)
    report_lines = [
        f"# ACL2 v118 Stage4-{STAGE_LABEL} Fresh R47 Policy",
        "",
        f"- decision: `{decision}`",
        f"- complete: `{complete}`",
        f"- action_fidelity: `{action_fidelity}`",
        f"- policy_gate: `{policy_gate}`",
        f"- global_goal_achieved: `{summary['global_goal_achieved']}`",
        "",
        "| seq | action | selected | selected ATE | rel vs default | control role | control ATE | selected - control |",
        "|---|---|---|---:|---:|---|---:|---:|",
    ]
    for row in comparison_rows:
        report_lines.append(
            f"| {row['seq']} | {row['policy_action']} | {row['selected_method']} | {row['selected_ate']} | "
            f"{row['selected_rel_vs_default']} | {row['control_role']} | {row['control_ate']} | {row['selected_minus_control_ate']} |"
        )
    report_lines += ["", "## Boundary", "", summary["boundary"]]
    report_path.write_text("\n".join(report_lines).rstrip() + "\n", encoding="utf-8")
    print(json.dumps(clean_json(summary), indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
