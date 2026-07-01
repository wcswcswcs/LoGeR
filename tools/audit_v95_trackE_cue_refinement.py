#!/usr/bin/env python3
"""Fail-forward Track E cue refinement audit for ACL2 v95.

This script searches fixed, training-free cue rules over the measured v94
action-surface rows. It separates method-safe cue atoms from diagnostic-only
upper-bound atoms, then compares every candidate with same-count and rotation
controls using the same measured runtime-probe effect columns.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from itertools import combinations
from pathlib import Path
from typing import Any


V95_ROOT = Path("results/acl2_v95tf_multiroute_semantic_memory_evidence_control")
V94_ROOT = Path("results/acl2_v94tf_semantic_gauge_failure_localization_causal_memory_control")
DEFAULT_EFFECT_ROWS = V94_ROOT / "phase6_object_source_action_surface/action_surface_effect_rows.csv"
DEFAULT_SELECTED_ROWS = V94_ROOT / "phase5_object_source_extension/selected_policy_rows.csv"
DEFAULT_OUT_DIR = V95_ROOT / "trackE_swa_transport_repair"


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def f(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def median(values: list[float]) -> float | None:
    finite = sorted(value for value in values if math.isfinite(value))
    if not finite:
        return None
    mid = len(finite) // 2
    if len(finite) % 2:
        return float(finite[mid])
    return float((finite[mid - 1] + finite[mid]) / 2.0)


def max_or_none(values: list[float]) -> float | None:
    finite = [value for value in values if math.isfinite(value)]
    return float(max(finite)) if finite else None


def quantile(values: list[float], q: float) -> float:
    finite = sorted(value for value in values if math.isfinite(value))
    if not finite:
        return float("nan")
    pos = (len(finite) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return finite[int(pos)]
    frac = pos - lo
    return finite[lo] * (1.0 - frac) + finite[hi] * frac


def stable_key(*parts: Any) -> float:
    digest = hashlib.sha256("|".join(str(part) for part in parts).encode("utf-8")).hexdigest()
    return int(digest[:12], 16) / float(16**12)


def rotate(mask: list[bool], amount: int) -> list[bool]:
    if not mask:
        return []
    amount %= len(mask)
    return mask[amount:] + mask[:amount]


def regime_rotate(rows: list[dict[str, Any]], mask: list[bool], amount: int) -> list[bool]:
    out = [False] * len(mask)
    by_seq: dict[str, list[int]] = {}
    for idx, row in enumerate(rows):
        by_seq.setdefault(str(row.get("seq")), []).append(idx)
    for indices in by_seq.values():
        values = rotate([mask[idx] for idx in indices], amount)
        for idx, value in zip(indices, values):
            out[idx] = value
    return out


def object_order_rotate(rows: list[dict[str, Any]], mask: list[bool], amount: int) -> list[bool]:
    order = sorted(range(len(rows)), key=lambda idx: (f(rows[idx].get("object_boundary_ratio")), idx))
    values = rotate([mask[idx] for idx in order], amount)
    out = [False] * len(mask)
    for idx, value in zip(order, values):
        out[idx] = value
    return out


def same_count_random(rows: list[dict[str, Any]], mask: list[bool]) -> list[bool]:
    out = [False] * len(mask)
    groups: dict[tuple[str, str], list[int]] = {}
    for idx, row in enumerate(rows):
        groups.setdefault((str(row.get("seq")), str(row.get("case_label_offline_only"))), []).append(idx)
    for (seq, label), indices in groups.items():
        count = sum(1 for idx in indices if mask[idx])
        ordered = sorted(indices, key=lambda idx: stable_key("v95_tracke_same_count", seq, label, rows[idx].get("pair_id")))
        for idx in ordered[: min(count, len(ordered))]:
            out[idx] = True
    return out


def merge_metadata(effect_rows: list[dict[str, str]], selected_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    selected_by_pair = {row.get("pair_id"): row for row in selected_rows}
    rows: list[dict[str, Any]] = []
    for row in effect_rows:
        if row.get("case_label_offline_only") not in {"bad", "good"}:
            continue
        out: dict[str, Any] = dict(row)
        meta = selected_by_pair.get(row.get("pair_id"), {})
        for key, value in meta.items():
            out.setdefault(key, value)
        for key in [
            "selected_policy_positive",
            "semantic_evidence_type",
            "boundary_global_cross_ratio",
            "boundary_new_id_ratio",
            "object_boundary_ratio",
            "same_global_object_id",
            "radio_boundary_mean",
            "carrier_error_merge_residual_after_abs",
            "carrier_error_abs_log_scale_jump_runtime",
        ]:
            if key in meta:
                out[key] = meta[key]
        rows.append(out)
    return rows


def bool_text(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def not_mask(mask: list[bool]) -> list[bool]:
    return [not value for value in mask]


def and_mask(*masks: list[bool]) -> list[bool]:
    return [all(values) for values in zip(*masks)]


def or_mask(*masks: list[bool]) -> list[bool]:
    return [any(values) for values in zip(*masks)]


def build_atoms(rows: list[dict[str, Any]]) -> tuple[dict[str, list[bool]], dict[str, list[bool]], dict[str, float]]:
    thresholds = {}
    for column in [
        "boundary_global_cross_ratio",
        "boundary_new_id_ratio",
        "object_boundary_ratio",
        "radio_boundary_mean",
        "carrier_error_merge_residual_after_abs",
        "carrier_error_abs_log_scale_jump_runtime",
    ]:
        values = [f(row.get(column)) for row in rows]
        for q in [0.50, 0.60, 0.75, 0.85, 0.90]:
            thresholds[f"{column}_q{int(q * 100)}"] = quantile(values, q)

    semantic_roles = sorted({str(row.get("semantic_evidence_type")) for row in rows if row.get("semantic_evidence_type")})
    method_atoms: dict[str, list[bool]] = {
        "OBJECT_SOURCE_POLICY": [bool_text(row.get("object_source_policy_positive")) or bool_text(row.get("selected_policy_positive")) for row in rows],
        "SAME_GLOBAL_OBJECT_FALSE": [str(row.get("same_global_object_id")).lower() == "false" for row in rows],
    }
    for role in semantic_roles:
        safe_name = role.replace("SEM_", "SEM_")
        method_atoms[safe_name] = [str(row.get("semantic_evidence_type")) == role for row in rows]
    for column in ["boundary_global_cross_ratio", "boundary_new_id_ratio", "object_boundary_ratio", "radio_boundary_mean"]:
        for q in [50, 60, 75, 85, 90]:
            thresh = thresholds[f"{column}_q{q}"]
            method_atoms[f"{column.upper()}_GE_Q{q}"] = [f(row.get(column)) >= thresh for row in rows]

    diagnostic_atoms = dict(method_atoms)
    for column in ["carrier_error_merge_residual_after_abs", "carrier_error_abs_log_scale_jump_runtime"]:
        for q in [50, 60, 75, 85, 90]:
            thresh = thresholds[f"{column}_q{q}"]
            diagnostic_atoms[f"{column.upper()}_GE_Q{q}"] = [f(row.get(column)) >= thresh for row in rows]
    for failure in sorted({str(row.get("failure_type_primary")) for row in rows if row.get("failure_type_primary")}):
        diagnostic_atoms[f"FAILURE_{failure}"] = [str(row.get("failure_type_primary")) == failure for row in rows]
    diagnostic_atoms["LABEL_BAD_UPPER_BOUND"] = [row.get("case_label_offline_only") == "bad" for row in rows]
    return method_atoms, diagnostic_atoms, thresholds


def candidate_masks(atoms: dict[str, list[bool]], max_or_terms: int = 2) -> dict[str, list[bool]]:
    candidates: dict[str, list[bool]] = {}
    names = sorted(atoms)
    for name in names:
        candidates[name] = atoms[name]
        if "SEM_MULTIMODE_UNSAFE" in atoms and name != "SEM_MULTIMODE_UNSAFE":
            candidates[f"{name}__AND_NOT_SEM_MULTIMODE"] = and_mask(atoms[name], not_mask(atoms["SEM_MULTIMODE_UNSAFE"]))
        if "SEM_LOWOBS_ABSTAIN" in atoms and name != "SEM_LOWOBS_ABSTAIN":
            candidates[f"{name}__AND_NOT_SEM_LOWOBS"] = and_mask(atoms[name], not_mask(atoms["SEM_LOWOBS_ABSTAIN"]))
        if "SEM_MULTIMODE_UNSAFE" in atoms and "SEM_LOWOBS_ABSTAIN" in atoms and name not in {"SEM_MULTIMODE_UNSAFE", "SEM_LOWOBS_ABSTAIN"}:
            candidates[f"{name}__AND_NOT_MULTIMODE_NOT_LOWOBS"] = and_mask(
                atoms[name],
                not_mask(atoms["SEM_MULTIMODE_UNSAFE"]),
                not_mask(atoms["SEM_LOWOBS_ABSTAIN"]),
            )
    for term_count in range(2, max_or_terms + 1):
        for combo in combinations(names, term_count):
            name = "_OR_".join(combo)
            candidates[name] = or_mask(*(atoms[item] for item in combo))
    return {name: mask for name, mask in candidates.items() if any(mask)}


def metric(rows: list[dict[str, Any]], mask: list[bool], name: str, scope: str) -> dict[str, Any]:
    selected = [idx for idx, value in enumerate(mask) if value]
    bad = [idx for idx in selected if rows[idx].get("case_label_offline_only") == "bad"]
    good = [idx for idx in selected if rows[idx].get("case_label_offline_only") == "good"]
    bad_i = [f(rows[idx].get("I_J_runtime_proxy")) for idx in bad]
    good_w = [f(rows[idx].get("W_good_runtime_proxy")) for idx in good]
    carrier_delta = [abs(f(rows[idx].get("carrier_state_delta"))) for idx in selected]
    trajectory_available = [
        math.isfinite(f(rows[idx].get("probe_curr_postmerge_sim3_rmse")))
        and math.isfinite(f(rows[idx].get("probe_curr_handoff_transfer_rmse")))
        for idx in selected
    ]
    return {
        "policy": name,
        "scope": scope,
        "selected_row_count": len(selected),
        "bad_rows": len(bad),
        "good_rows": len(good),
        "bad_recall_on_measured_labelled": len(bad) / max(sum(1 for row in rows if row.get("case_label_offline_only") == "bad"), 1),
        "good_FPR_on_measured_labelled": len(good) / max(sum(1 for row in rows if row.get("case_label_offline_only") == "good"), 1),
        "sequence_coverage": len({rows[idx].get("seq") for idx in selected}),
        "bad_median_I_J_runtime_proxy": median(bad_i),
        "bad_min_I_J_runtime_proxy": min(bad_i) if bad_i else None,
        "bad_negative_improvement_rows": sum(1 for value in bad_i if math.isfinite(value) and value < 0),
        "good_median_worsen_runtime_proxy": median(good_w),
        "good_max_worsen_runtime_proxy": max_or_none(good_w),
        "good_worsen_gt_0p02_rows": sum(1 for value in good_w if math.isfinite(value) and value > 0.02),
        "carrier_state_delta_nonzero_rows": sum(1 for value in carrier_delta if math.isfinite(value) and value > 1e-9),
        "trajectory_rows_complete": all(trajectory_available) if selected else False,
        "bad_hits": ",".join(str(rows[idx].get("pair_id")) for idx in bad),
        "good_hits": ",".join(str(rows[idx].get("pair_id")) for idx in good),
    }


def evaluate_candidate(rows: list[dict[str, Any]], mask: list[bool], name: str, scope: str) -> dict[str, Any]:
    actual = metric(rows, mask, name, scope)
    controls = [
        metric(rows, same_count_random(rows, mask), f"{name}__CONTROL_same_count_random", scope),
        metric(rows, object_order_rotate(rows, mask, 5), f"{name}__CONTROL_object_order_rot5", scope),
        metric(rows, regime_rotate(rows, mask, 1), f"{name}__CONTROL_regime_rot1", scope),
        metric(rows, regime_rotate(rows, mask, 2), f"{name}__CONTROL_regime_rot2", scope),
        metric(rows, rotate(mask, 7), f"{name}__CONTROL_same_count_rot7", scope),
    ]
    best_control_bad = max(
        [row["bad_median_I_J_runtime_proxy"] for row in controls if row["bad_median_I_J_runtime_proxy"] is not None],
        default=None,
    )
    actual_bad = actual.get("bad_median_I_J_runtime_proxy")
    actual_minus_best_control = (
        float(actual_bad - best_control_bad) if actual_bad is not None and best_control_bad is not None else None
    )
    good_median = actual.get("good_median_worsen_runtime_proxy")
    good_max = actual.get("good_max_worsen_runtime_proxy")
    gates = {
        "bad_runtime_proxy_improvement_gate": actual_bad is not None and actual_bad >= 0.05,
        "good_median_protection_gate": good_median is not None and good_median <= 0.02,
        "good_catastrophic_worsen_absent": good_max is not None and good_max <= 0.05,
        "sequence_coverage_gate": int(actual.get("sequence_coverage") or 0) >= 3,
        "trajectory_rows_complete_gate": bool(actual.get("trajectory_rows_complete")),
        "carrier_state_delta_gate": int(actual.get("carrier_state_delta_nonzero_rows") or 0) > 0,
        "beats_measured_selection_controls_gate": actual_minus_best_control is not None and actual_minus_best_control >= 0.05,
        "non_empty_bad_and_good_context_gate": actual["bad_rows"] > 0 and actual["selected_row_count"] > 0,
    }
    gate_pass = all(gates.values())
    out = {
        **actual,
        "best_control_bad_median_I_J_runtime_proxy": best_control_bad,
        "actual_minus_best_control": actual_minus_best_control,
        **gates,
        "candidate_gate_pass": gate_pass,
    }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--effect-rows", type=Path, default=DEFAULT_EFFECT_ROWS)
    parser.add_argument("--selected-rows", type=Path, default=DEFAULT_SELECTED_ROWS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    rows = merge_metadata(read_csv_rows(args.effect_rows), read_csv_rows(args.selected_rows))
    method_atoms, diagnostic_atoms, thresholds = build_atoms(rows)
    method_candidates = candidate_masks(method_atoms)
    diagnostic_candidates = candidate_masks(diagnostic_atoms)

    metrics: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for name, mask in method_candidates.items():
        key = ("method_safe", name)
        if key not in seen:
            metrics.append(evaluate_candidate(rows, mask, name, "method_safe"))
            seen.add(key)
    for name, mask in diagnostic_candidates.items():
        key = ("diagnostic_only", name)
        if key not in seen:
            metrics.append(evaluate_candidate(rows, mask, name, "diagnostic_only"))
            seen.add(key)
    metrics.sort(
        key=lambda row: (
            bool(row["candidate_gate_pass"]),
            row["scope"] == "method_safe",
            f(row.get("actual_minus_best_control")),
            f(row.get("bad_median_I_J_runtime_proxy")),
            -f(row.get("good_max_worsen_runtime_proxy")),
            f(row.get("bad_recall_on_measured_labelled")),
            -f(row.get("good_FPR_on_measured_labelled")),
        ),
        reverse=True,
    )
    method_rows = [row for row in metrics if row["scope"] == "method_safe"]
    diag_rows = [row for row in metrics if row["scope"] == "diagnostic_only"]
    method_passing = [row for row in method_rows if row["candidate_gate_pass"]]
    diag_passing = [row for row in diag_rows if row["candidate_gate_pass"]]
    best_method = method_rows[0] if method_rows else {}
    best_diag = diag_rows[0] if diag_rows else {}
    summary = {
        "phase": "v95_trackE_fail_forward_cue_refinement",
        "effect_rows": str(args.effect_rows),
        "selected_rows": str(args.selected_rows),
        "measured_labelled_rows": len(rows),
        "method_candidate_count": len(method_rows),
        "diagnostic_candidate_count": len(diag_rows),
        "method_safe_passing_count": len(method_passing),
        "diagnostic_only_passing_count": len(diag_passing),
        "method_safe_gate_pass": bool(method_passing),
        "diagnostic_upper_bound_gate_pass": bool(diag_passing),
        "best_method_safe": best_method,
        "best_diagnostic_only": best_diag,
        "thresholds": thresholds,
        "runtime_action_allowed": bool(method_passing),
        "blocker": "" if method_passing else "no_method_safe_candidate_beats_measured_controls",
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "candidate_policy_metrics.csv", metrics)
    write_csv(args.out_dir / "method_safe_candidate_metrics.csv", method_rows)
    write_csv(args.out_dir / "diagnostic_only_candidate_metrics.csv", diag_rows)
    write_json(args.out_dir / "cue_refinement_summary.json", summary)
    write_json(args.out_dir / "summary.json", summary)
    write_csv(
        args.out_dir / "gate_checks.csv",
        [
            {"check": "method_safe_candidate_passes_measured_controls", "pass": bool(method_passing), "value": len(method_passing), "required": ">0"},
            {"check": "diagnostic_upper_bound_passes_measured_controls", "pass": bool(diag_passing), "value": len(diag_passing), "required": "diagnostic only"},
        ],
    )
    write_csv(args.out_dir / "rows.csv", metrics)
    write_csv(args.out_dir / "visual_manifest.csv", [])
    write_text(
        args.out_dir / "failure_report.md",
        "No method-safe cue candidate beat measured controls." if not method_passing else "At least one method-safe candidate passed diagnostic gates; replay/promotion still required.",
    )
    write_text(
        args.out_dir / "what_would_have_to_be_true_to_pass.md",
        "A pass requires method-safe cue selection with bad runtime proxy improvement, good protection, sequence coverage, carrier delta, complete trajectory rows, and actual-minus-best-control >= 0.05.",
    )
    write_text(
        args.out_dir / "analysis.md",
        f"""
# Track E Cue Refinement Audit

- measured_labelled_rows: `{len(rows)}`
- method_candidate_count: `{len(method_rows)}`
- diagnostic_candidate_count: `{len(diag_rows)}`
- method_safe_passing_count: `{len(method_passing)}`
- diagnostic_only_passing_count: `{len(diag_passing)}`
- best_method_safe: `{best_method.get('policy')}`
- best_method_safe_actual_minus_best_control: `{best_method.get('actual_minus_best_control')}`
- best_method_safe_bad_median_I_J: `{best_method.get('bad_median_I_J_runtime_proxy')}`
- best_method_safe_good_max_worsen: `{best_method.get('good_max_worsen_runtime_proxy')}`
- blocker: `{summary['blocker']}`

This audit is still training-free: rules are fixed combinations of semantic/object-source/trace atoms. Diagnostic-only rows that use failure labels or carrier error thresholds are reported separately and are not method-safe runtime candidates.
""",
    )
    print(f"method_candidate_count={len(method_rows)}")
    print(f"diagnostic_candidate_count={len(diag_rows)}")
    print(f"method_safe_passing_count={len(method_passing)}")
    print(f"diagnostic_only_passing_count={len(diag_passing)}")
    print(f"best_method_safe={best_method.get('policy')}")
    print(f"best_method_safe_actual_minus_best_control={best_method.get('actual_minus_best_control')}")
    print(f"runtime_action_allowed={summary['runtime_action_allowed']}")


if __name__ == "__main__":
    main()
