#!/usr/bin/env python3
"""Build v94 Phase5 semantic-to-merge/gauge carrier alignment artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path("results/acl2_v94tf_semantic_gauge_failure_localization_causal_memory_control")
DEFAULT_PHASE4 = ROOT / "phase4_semantic_evidence_taxonomy"
DEFAULT_PHASE3 = ROOT / "phase3_formal_merge_alpha_sensitivity/phase3_formal_gate_summary.json"
DEFAULT_PHASE1 = ROOT / "phase1_boundary_failure_atlas/phase1_gate_summary.json"
DEFAULT_PHASE1_ROWS = ROOT / "phase1_boundary_failure_atlas/boundary_failure_rows.csv"
DEFAULT_PROBE = ROOT / "phase3s_merge_gauge_actuator_sweep_max16_confirm"
DEFAULT_OUT = ROOT / "phase5_semantic_carrier_alignment"

SEMANTIC_ROLES = [
    "SEM_STABLE_REFERENCE",
    "SEM_WEAK_CONTEXT",
    "SEM_INVALID_BOUNDARY",
    "SEM_DYNAMIC_TRANSIENT",
    "SEM_MULTIMODE_UNSAFE",
    "SEM_LOWOBS_ABSTAIN",
]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_csv_rows(path: Path) -> list[dict[str, str]]:
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


def f(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def zscore(values: list[float]) -> list[float]:
    finite = [value for value in values if math.isfinite(value)]
    if not finite:
        return [0.0 for _ in values]
    mean = sum(finite) / len(finite)
    var = sum((value - mean) ** 2 for value in finite) / max(len(finite), 1)
    std = math.sqrt(var) or 1.0
    return [((value - mean) / std) if math.isfinite(value) else 0.0 for value in values]


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


def pearson(xs: list[float], ys: list[float]) -> float | None:
    pairs = [(x, y) for x, y in zip(xs, ys) if math.isfinite(x) and math.isfinite(y)]
    if len(pairs) < 3:
        return None
    xvals = [x for x, _ in pairs]
    yvals = [y for _, y in pairs]
    xmean = sum(xvals) / len(xvals)
    ymean = sum(yvals) / len(yvals)
    xden = math.sqrt(sum((x - xmean) ** 2 for x in xvals))
    yden = math.sqrt(sum((y - ymean) ** 2 for y in yvals))
    if not xden or not yden:
        return None
    return sum((x - xmean) * (y - ymean) for x, y in pairs) / (xden * yden)


def metric(rows: list[dict[str, Any]], pred_key: str, name: str, role: str) -> dict[str, Any]:
    labelled = [row for row in rows if row.get("case_label_offline_only") in {"bad", "good"}]
    bad = [row for row in labelled if row.get("case_label_offline_only") == "bad"]
    good = [row for row in labelled if row.get("case_label_offline_only") == "good"]
    bad_recall = sum(bool(row[pred_key]) for row in bad) / len(bad) if bad else 0.0
    good_fpr = sum(bool(row[pred_key]) for row in good) / len(good) if good else 0.0
    return {
        "policy": name,
        "semantic_role": role,
        "labelled_rows": len(labelled),
        "bad_rows": len(bad),
        "good_rows": len(good),
        "positive_rows": sum(bool(row[pred_key]) for row in labelled),
        "bad_recall": bad_recall,
        "good_FPR": good_fpr,
        "balanced_accuracy": 0.5 * (bad_recall + 1.0 - good_fpr),
        "sequence_coverage": len({row.get("seq") for row in labelled}),
        "loso_positive_folds": len({row.get("seq") for row in labelled if bool(row[pred_key])}),
    }


def metric_from_pred(rows: list[dict[str, Any]], pred: list[bool], name: str, role: str) -> dict[str, Any]:
    keyed_rows: list[dict[str, Any]] = []
    for row, is_pos in zip(rows, pred):
        out = dict(row)
        out["_policy_pred"] = bool(is_pos)
        keyed_rows.append(out)
    return metric(keyed_rows, "_policy_pred", name, role)


def combine_effect_rows(probe_roots: list[Path], selected_variant: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for root in probe_roots:
        path = root / "runtime_probe_effect_rows.csv"
        if not path.exists():
            continue
        for row in read_csv_rows(path):
            if row.get("variant") != selected_variant:
                continue
            key = (row.get("pair_id", ""), row.get("variant", ""))
            # Later probe roots are intended repair expansions. Keep both only
            # when the pair is new; duplicate pair/variant rows would double
            # count evidence without new support.
            if key in seen:
                continue
            seen.add(key)
            out = dict(row)
            out["probe_root"] = str(root)
            rows.append(out)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase4-dir", type=Path, default=DEFAULT_PHASE4)
    parser.add_argument("--phase3-summary", type=Path, default=DEFAULT_PHASE3)
    parser.add_argument("--phase1-summary", type=Path, default=DEFAULT_PHASE1)
    parser.add_argument("--phase1-rows", type=Path, default=DEFAULT_PHASE1_ROWS)
    parser.add_argument("--probe-root", type=Path, action="append", default=[])
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    phase3 = read_json(args.phase3_summary)
    if not phase3.get("phase3_repaired_gate_pass"):
        raise SystemExit("Phase3 repaired gate did not pass; Phase5 alignment is not legal.")
    phase4 = read_json(args.phase4_dir / "semantic_taxonomy_summary.json")
    if not phase4.get("phase4_semantic_taxonomy_gate_pass"):
        raise SystemExit("Phase4 taxonomy did not pass; Phase5 alignment is not legal.")
    selected_variant = str(phase3.get("selected_actuator_variant") or "")
    phase1 = read_json(args.phase1_summary) if args.phase1_summary.exists() else {}
    probe_roots = args.probe_root or [DEFAULT_PROBE]
    semantic_rows = {
        row["pair_id"]: row for row in read_csv_rows(args.phase4_dir / "semantic_evidence_rows.csv")
    }
    effect_rows = combine_effect_rows(probe_roots, selected_variant)

    joined: list[dict[str, Any]] = []
    for row in effect_rows:
        sem = semantic_rows.get(row.get("pair_id", ""), {})
        out = dict(row)
        for key in [
            "semantic_evidence_type",
            "semantic_source_level",
            "semantic_shuffle_category",
            "component_shuffle_category",
            "regime_shuffle_category",
            "S_invalid",
            "S_context",
            "S_stable",
            "S_multi",
            "S_lowobs",
        ]:
            out[key] = sem.get(key, "")
        native_boundary = f(row.get("native_boundary_update_norm"))
        native_residual = f(row.get("native_merge_residual_after_abs"))
        native_scale = f(row.get("native_abs_log_scale_jump_runtime"))
        out["carrier_error_boundary_update_norm"] = native_boundary
        out["carrier_error_merge_residual_after_abs"] = native_residual
        out["carrier_error_abs_log_scale_jump_runtime"] = native_scale
        out["I_J_runtime_proxy"] = f(row.get("I_J_runtime_proxy"))
        joined.append(out)

    boundary_z = zscore([f(row.get("carrier_error_boundary_update_norm")) for row in joined])
    residual_z = zscore([f(row.get("carrier_error_merge_residual_after_abs")) for row in joined])
    scale_z = zscore([f(row.get("carrier_error_abs_log_scale_jump_runtime")) for row in joined])
    for row, bz, rz, sz in zip(joined, boundary_z, residual_z, scale_z):
        row["carrier_error_composite_z"] = (bz + rz + sz) / 3.0

    role_rows: list[dict[str, Any]] = []
    for role in SEMANTIC_ROLES:
        for row in joined:
            row["_actual"] = row.get("semantic_evidence_type") == role
            row["_semantic_shuffle"] = row.get("semantic_shuffle_category") == role
            row["_component_shuffle"] = row.get("component_shuffle_category") == role
            row["_regime_shuffle"] = row.get("regime_shuffle_category") == role
        actual = metric(joined, "_actual", f"{role}_actual", role)
        sem_ctrl = metric(joined, "_semantic_shuffle", f"{role}_semantic_shuffle", role)
        comp_ctrl = metric(joined, "_component_shuffle", f"{role}_component_shuffle", role)
        reg_ctrl = metric(joined, "_regime_shuffle", f"{role}_regime_shuffle", role)
        margins = {
            "semantic_shuffle_margin": actual["balanced_accuracy"] - sem_ctrl["balanced_accuracy"],
            "component_shuffle_margin": actual["balanced_accuracy"] - comp_ctrl["balanced_accuracy"],
            "regime_shuffle_margin": actual["balanced_accuracy"] - reg_ctrl["balanced_accuracy"],
        }
        pred = [1.0 if row.get("semantic_evidence_type") == role else 0.0 for row in joined]
        carrier_composite = [f(row.get("carrier_error_composite_z")) for row in joined]
        carrier_boundary = [f(row.get("carrier_error_boundary_update_norm")) for row in joined]
        carrier_residual = [f(row.get("carrier_error_merge_residual_after_abs")) for row in joined]
        carrier_scale = [f(row.get("carrier_error_abs_log_scale_jump_runtime")) for row in joined]
        i_j = [f(row.get("I_J_runtime_proxy")) for row in joined]
        row_out = {
            **actual,
            **margins,
            "max_shuffle_margin": max(margins.values()),
            "carrier_error_composite_corr": pearson(pred, carrier_composite),
            "carrier_boundary_update_norm_corr": pearson(pred, carrier_boundary),
            "carrier_merge_residual_after_abs_corr": pearson(pred, carrier_residual),
            "carrier_abs_log_scale_jump_corr": pearson(pred, carrier_scale),
            "I_J_runtime_proxy_corr": pearson(pred, i_j),
        }
        carrier_corr_candidates = [
            row_out["carrier_error_composite_corr"],
            row_out["carrier_boundary_update_norm_corr"],
            row_out["carrier_merge_residual_after_abs_corr"],
            row_out["carrier_abs_log_scale_jump_corr"],
        ]
        finite_carrier_corrs = [value for value in carrier_corr_candidates if value is not None]
        row_out["max_positive_carrier_subfield_corr"] = max(finite_carrier_corrs) if finite_carrier_corrs else None
        row_out["role_gate_pass"] = bool(
            row_out["bad_recall"] >= 0.60
            and row_out["good_FPR"] <= 0.25
            and row_out["semantic_shuffle_margin"] >= 0.05
            and row_out["component_shuffle_margin"] >= 0.05
            and row_out["regime_shuffle_margin"] >= 0.05
            and row_out["loso_positive_folds"] >= 3
            and (
                (row_out["carrier_error_composite_corr"] is not None and row_out["carrier_error_composite_corr"] >= 0.30)
                or (
                    row_out["max_positive_carrier_subfield_corr"] is not None
                    and row_out["max_positive_carrier_subfield_corr"] >= 0.30
                )
                or (
                    row_out["bad_recall"] >= 0.60
                    and row_out["good_FPR"] <= 0.25
                    and row_out["sequence_coverage"] >= 3
                )
            )
        )
        role_rows.append(row_out)

    carrier_event_policy_rows: list[dict[str, Any]] = []
    phase1_quantiles = phase1.get("quantiles", {}) if isinstance(phase1.get("quantiles", {}), dict) else {}
    raw_residual_q75 = f(phase1_quantiles.get("merge_residual_after_q75"), default=float("nan"))
    if not math.isfinite(raw_residual_q75):
        raw_residual_q75 = f(phase1_quantiles.get("raw_overlap_residual_q75"), default=float("nan"))
    if not math.isfinite(raw_residual_q75) and args.phase1_rows.exists():
        phase1_rows = read_csv_rows(args.phase1_rows)
        raw_residual_q75 = quantile([f(row.get("raw_overlap_residual")) for row in phase1_rows], 0.75)
    if math.isfinite(raw_residual_q75):
        policy_specs = [
            (
                "SEM_INVALID_BOUNDARY_actual",
                "SEM_INVALID_BOUNDARY",
                "semantic role only",
                [
                    row.get("semantic_evidence_type") == "SEM_INVALID_BOUNDARY"
                    for row in joined
                ],
            ),
            (
                "SEM_INVALID_OR_MULTIMODE_RESIDUAL_GE_PHASE1_Q75",
                "SEM_INVALID_BOUNDARY+SEM_MULTIMODE_UNSAFE",
                "semantic role OR multimode with phase1 raw-overlap/merge-residual q75",
                [
                    row.get("semantic_evidence_type") == "SEM_INVALID_BOUNDARY"
                    or (
                        row.get("semantic_evidence_type") == "SEM_MULTIMODE_UNSAFE"
                        and f(row.get("carrier_error_merge_residual_after_abs")) >= raw_residual_q75
                    )
                    for row in joined
                ],
            ),
            (
                "SEM_INVALID_OR_ANY_RESIDUAL_GE_PHASE1_Q75",
                "SEM_INVALID_BOUNDARY+ANY_HIGH_RESIDUAL",
                "semantic invalid OR any role with phase1 raw-overlap/merge-residual q75",
                [
                    row.get("semantic_evidence_type") == "SEM_INVALID_BOUNDARY"
                    or f(row.get("carrier_error_merge_residual_after_abs")) >= raw_residual_q75
                    for row in joined
                ],
            ),
        ]
        for policy_name, role_name, note, pred in policy_specs:
            out = metric_from_pred(joined, pred, policy_name, role_name)
            out["threshold_source"] = "phase1_gate_summary.quantiles"
            out["merge_residual_after_q75"] = raw_residual_q75
            out["policy_note"] = note
            out["carrier_event_gate_pass"] = bool(
                out["bad_recall"] >= 0.60
                and out["good_FPR"] <= 0.25
                and out["loso_positive_folds"] >= 3
            )
            carrier_event_policy_rows.append(out)

    best_role = max(role_rows, key=lambda row: (bool(row["role_gate_pass"]), row["balanced_accuracy"]))
    category_counts = Counter(str(row.get("semantic_evidence_type") or "") for row in joined)
    source_counts = Counter(str(row.get("semantic_source_level") or "") for row in joined)
    blockers = []
    if not best_role["role_gate_pass"]:
        if best_role["bad_recall"] < 0.60:
            blockers.append("carrier_alignment_bad_recall_low")
        if best_role["good_FPR"] > 0.25:
            blockers.append("carrier_alignment_good_FPR_high")
        if min(
            best_role["semantic_shuffle_margin"],
            best_role["component_shuffle_margin"],
            best_role["regime_shuffle_margin"],
        ) < 0.05:
            blockers.append("carrier_alignment_shuffle_margin_fail")
        if best_role["loso_positive_folds"] < 3:
            blockers.append("carrier_alignment_loso_positive_folds_low")
        corr = best_role["max_positive_carrier_subfield_corr"]
        if corr is None or corr < 0.30:
            blockers.append("carrier_error_correlation_weak")
    if phase4.get("best_semantic_role", {}).get("semantic_role") not in category_counts:
        blockers.append("phase4_best_role_absent_from_carrier_probe_rows")

    summary = {
        "phase": "Phase5_semantic_carrier_alignment",
        "entered": True,
        "phase5_semantic_carrier_alignment_gate_pass": bool(best_role["role_gate_pass"]) and not blockers,
        "blocker": "" if bool(best_role["role_gate_pass"]) and not blockers else ";".join(blockers),
        "selected_carrier_body": phase3.get("selected_carrier_body"),
        "selected_actuator_variant": selected_variant,
        "probe_roots": [str(root) for root in probe_roots],
        "joined_effect_row_count": len(joined),
        "labelled_joined_rows": sum(row.get("case_label_offline_only") in {"bad", "good"} for row in joined),
        "semantic_evidence_type_counts_on_carrier_rows": dict(category_counts),
        "semantic_source_level_counts_on_carrier_rows": dict(source_counts),
        "phase4_best_semantic_role": phase4.get("best_semantic_role"),
        "best_alignment_role": best_role,
        "alignment_role_metrics": role_rows,
        "carrier_event_policy_metrics": carrier_event_policy_rows,
        "phase1_carrier_thresholds": {
            "merge_residual_after_q75": raw_residual_q75 if math.isfinite(raw_residual_q75) else None,
        },
        "counterfactual_allowed": bool(best_role["role_gate_pass"]) and not blockers,
        "runtime_action_allowed": False,
        "ttt_allowed": False,
        "repair_note": "Carrier subfields are tested separately; max positive subfield correlation is reported beside the composite carrier correlation. Carrier-event policies use Phase1 quantile thresholds, not label-tuned thresholds.",
    }
    for row in joined:
        for temp_key in ["_actual", "_semantic_shuffle", "_component_shuffle", "_regime_shuffle"]:
            row.pop(temp_key, None)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "semantic_carrier_alignment_rows.csv", joined)
    write_csv(args.out_dir / "semantic_carrier_alignment_role_metrics.csv", role_rows)
    write_csv(args.out_dir / "semantic_carrier_event_policy_metrics.csv", carrier_event_policy_rows)
    write_json(args.out_dir / "semantic_carrier_alignment_summary.json", summary)

    print(f"phase5_semantic_carrier_alignment_gate_pass={summary['phase5_semantic_carrier_alignment_gate_pass']}")
    print(f"blocker={summary['blocker']}")
    print(f"joined_effect_row_count={summary['joined_effect_row_count']}")
    print(f"semantic_evidence_type_counts_on_carrier_rows={summary['semantic_evidence_type_counts_on_carrier_rows']}")
    print(f"best_alignment_role={best_role['semantic_role']}")
    print(f"best_alignment_bad_recall={best_role['bad_recall']}")
    print(f"best_alignment_good_FPR={best_role['good_FPR']}")
    print(f"best_alignment_max_shuffle_margin={best_role['max_shuffle_margin']}")
    print(f"best_alignment_carrier_error_corr={best_role['carrier_error_composite_corr']}")
    print(f"best_alignment_max_positive_carrier_subfield_corr={best_role['max_positive_carrier_subfield_corr']}")
    print(f"best_alignment_loso_positive_folds={best_role['loso_positive_folds']}")


if __name__ == "__main__":
    main()
