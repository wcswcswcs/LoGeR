#!/usr/bin/env python3
"""Trace-level counterfactual check for the v94 object-source Phase5 branch."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path("results/acl2_v94tf_semantic_gauge_failure_localization_causal_memory_control")
V93_ROOT = Path("results/acl2_v93tf_semantic_object_identity_merge_gauge_boundary_carrier")
V93_OBJECT_ROWS = V93_ROOT / "phase1_object_identity_row_join/object_identity_row_join.csv"
V93_CF_EFFECT_ROWS = V93_ROOT / "phase5_merge_gauge_counterfactual_upper_bound/counterfactual_cf6_effect_rows.csv"

POSITIVE_STATES = {
    "RESET_RISK",
    "DELAY",
    "REJECT",
    "UPDATE_OBJECT_GAUGE",
    "REJECT_OBJECT_CONFLICT",
    "DELAY_COMMIT",
    "GEOMETRY_RISK",
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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


def finite(value: Any) -> bool:
    return isinstance(value, (float, int)) and math.isfinite(float(value))


def median(values: list[float]) -> float | None:
    finite_values = sorted(value for value in values if math.isfinite(value))
    if not finite_values:
        return None
    mid = len(finite_values) // 2
    if len(finite_values) % 2:
        return float(finite_values[mid])
    return float((finite_values[mid - 1] + finite_values[mid]) / 2.0)


def mean(values: list[float]) -> float | None:
    finite_values = [value for value in values if math.isfinite(value)]
    if not finite_values:
        return None
    return float(sum(finite_values) / len(finite_values))


def max_or_none(values: list[float]) -> float | None:
    finite_values = [value for value in values if math.isfinite(value)]
    if not finite_values:
        return None
    return float(max(finite_values))


def quantile(values: list[float], q: float) -> float:
    finite_values = sorted(value for value in values if math.isfinite(value))
    if not finite_values:
        return float("nan")
    pos = (len(finite_values) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return finite_values[int(pos)]
    frac = pos - lo
    return finite_values[lo] * (1.0 - frac) + finite_values[hi] * frac


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
        quality = str(row.get("quality_type") or row.get("base_case_type"))
        groups.setdefault((str(row.get("seq")), quality), []).append(idx)
    for (seq, quality), indices in groups.items():
        count = sum(1 for idx in indices if mask[idx])
        if count <= 0:
            continue
        ordered = sorted(indices, key=lambda idx: stable_key("v94_object_phase6_same_count", seq, quality, rows[idx].get("pair_id")))
        for idx in ordered[: min(count, len(ordered))]:
            out[idx] = True
    return out


def bool_from_text(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def build_atoms(rows: list[dict[str, Any]], thresholds: dict[str, float]) -> dict[str, list[bool]]:
    return {
        "GLOBAL_CROSS_GE_Q75": [
            f(row.get("boundary_global_cross_ratio")) >= thresholds["boundary_global_cross_ratio_q75"] for row in rows
        ],
        "GLOBAL_NEW_GE_Q75": [
            f(row.get("boundary_new_id_ratio")) >= thresholds["boundary_new_id_ratio_q75"] for row in rows
        ],
        "OBJ_BOUNDARY_GE_Q75": [
            f(row.get("object_boundary_ratio")) >= thresholds["object_boundary_ratio_q75"] for row in rows
        ],
        "RADIO_BOUNDARY_GE_Q75": [
            f(row.get("radio_boundary_mean")) >= thresholds["radio_boundary_mean_q75"] for row in rows
        ],
        "SEM_INVALID": [row.get("semantic_evidence_type") == "SEM_INVALID_BOUNDARY" for row in rows],
        "SEM_LOWOBS": [row.get("semantic_evidence_type") == "SEM_LOWOBS_ABSTAIN" for row in rows],
        "SEM_WEAK_CONTEXT": [row.get("semantic_evidence_type") == "SEM_WEAK_CONTEXT" for row in rows],
        "V93_CF6_FULL_OBJECT_POLICY": [bool_from_text(row.get("cf6_action")) for row in rows],
        "V93_P5_POSITIVE_OBJECT_POLICY": [str(row.get("p5_combined_object_policy")) in POSITIVE_STATES for row in rows],
    }


def apply_policy(policy: str, atoms: dict[str, list[bool]]) -> list[bool]:
    parts = policy.split("_OR_")
    missing = [part for part in parts if part not in atoms]
    if missing:
        raise ValueError(f"Unsupported policy atoms for {policy}: {missing}")
    return [any(values) for values in zip(*(atoms[part] for part in parts))]


def ratio(numerator: float | None, denominator: float | None) -> float | None:
    if not finite(numerator):
        return None
    scale = max(abs(float(denominator or 0.0)), 1e-9)
    return float(numerator) / scale


def counterfactual_metrics(rows: list[dict[str, Any]], mask: list[bool], family: str) -> dict[str, Any]:
    labelled_indices = [idx for idx, row in enumerate(rows) if row.get("base_case_type") in {"bad", "good"}]
    bad_indices = [idx for idx, row in enumerate(rows) if row.get("base_case_type") == "bad"]
    good_indices = [idx for idx, row in enumerate(rows) if row.get("base_case_type") == "good"]

    native_delta = [f(row.get("native_merge_residual_delta")) for row in rows]
    native_update = [f(row.get("native_boundary_update_norm")) for row in rows]
    cf_delta = [0.0 if selected else value for selected, value in zip(mask, native_delta)]
    cf_update = [0.0 if selected else value for selected, value in zip(mask, native_update)]
    residual_improvement = [before - after for before, after in zip(native_delta, cf_delta)]
    update_reduction = [before - after for before, after in zip(native_update, cf_update)]
    residual_worsen = [after - before for before, after in zip(native_delta, cf_delta)]

    bad_native_med = median([native_delta[idx] for idx in bad_indices])
    bad_cf_med = median([cf_delta[idx] for idx in bad_indices])
    bad_improve_med = median([residual_improvement[idx] for idx in bad_indices])
    good_worsen_med = median([residual_worsen[idx] for idx in good_indices])
    good_worsen_max = max_or_none([residual_worsen[idx] for idx in good_indices])
    bad_scale = median([abs(native_delta[idx]) for idx in bad_indices])
    good_scale = median([abs(native_delta[idx]) for idx in good_indices])
    action_labelled = [idx for idx in labelled_indices if mask[idx]]
    action_bad = [idx for idx in bad_indices if mask[idx]]
    action_good = [idx for idx in good_indices if mask[idx]]
    bad_ratio = ratio(bad_improve_med, bad_scale)
    good_ratio = ratio(good_worsen_med, good_scale)
    good_max_ratio = ratio(good_worsen_max, good_scale)
    result = {
        "family": family,
        "action_row_count": sum(1 for value in mask if value),
        "labelled_action_row_count": len(action_labelled),
        "bad_rows": len(bad_indices),
        "good_rows": len(good_indices),
        "bad_action_rows": len(action_bad),
        "good_action_rows": len(action_good),
        "bad_action_recall": len(action_bad) / len(bad_indices) if bad_indices else 0.0,
        "good_action_FPR": len(action_good) / len(good_indices) if good_indices else 0.0,
        "sequence_coverage": len({rows[idx].get("seq") for idx in action_labelled}),
        "bad_native_merge_residual_delta_median": bad_native_med,
        "bad_counterfactual_merge_residual_delta_median": bad_cf_med,
        "bad_median_residual_improvement": bad_improve_med,
        "bad_median_residual_improvement_ratio": bad_ratio,
        "good_median_residual_worsen": good_worsen_med,
        "good_median_residual_worsen_ratio": good_ratio,
        "good_max_residual_worsen": good_worsen_max,
        "good_max_residual_worsen_ratio": good_max_ratio,
        "bad_mean_boundary_update_norm_reduction": mean([update_reduction[idx] for idx in bad_indices]),
        "good_mean_boundary_update_norm_reduction": mean([update_reduction[idx] for idx in good_indices]),
        "bad_hits": ",".join(str(rows[idx].get("pair_id")) for idx in action_bad),
        "good_hits": ",".join(str(rows[idx].get("pair_id")) for idx in action_good),
        "counterfactual_model": "trace_hold_sets_selected_merge_residual_delta_and_boundary_update_norm_to_zero",
        "actual_runtime_trajectory_counterfactual_available": False,
    }
    return result


def effect_rows(rows: list[dict[str, Any]], mask: list[bool], policy: str) -> list[dict[str, Any]]:
    out_rows: list[dict[str, Any]] = []
    for row, selected in zip(rows, mask):
        native_delta = f(row.get("native_merge_residual_delta"))
        native_update = f(row.get("native_boundary_update_norm"))
        cf_delta = 0.0 if selected else native_delta
        cf_update = 0.0 if selected else native_update
        improvement = native_delta - cf_delta
        out_rows.append(
            {
                "pair_id": row.get("pair_id"),
                "seq": row.get("seq"),
                "prev_chunk": row.get("prev_chunk"),
                "curr_chunk": row.get("curr_chunk"),
                "base_case_type": row.get("base_case_type"),
                "quality_type": row.get("quality_type"),
                "selected_policy": policy,
                "policy_action": selected,
                "semantic_evidence_type": row.get("semantic_evidence_type"),
                "boundary_global_cross_ratio": row.get("boundary_global_cross_ratio"),
                "boundary_new_id_ratio": row.get("boundary_new_id_ratio"),
                "object_boundary_ratio": row.get("object_boundary_ratio"),
                "radio_boundary_mean": row.get("radio_boundary_mean"),
                "native_merge_residual_delta": native_delta if math.isfinite(native_delta) else "",
                "cf_merge_residual_delta": cf_delta if math.isfinite(cf_delta) else "",
                "merge_residual_improvement": improvement if math.isfinite(improvement) else "",
                "native_boundary_update_norm": native_update if math.isfinite(native_update) else "",
                "cf_boundary_update_norm": cf_update if math.isfinite(cf_update) else "",
                "residual_effect_sign": (
                    "improves" if math.isfinite(improvement) and improvement > 0 else
                    "worsens" if math.isfinite(improvement) and improvement < 0 else
                    "neutral"
                ),
                "trace_path": row.get("trace_path"),
            }
        )
    return out_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--object-identity-rows", type=Path, default=V93_OBJECT_ROWS)
    parser.add_argument("--v93-counterfactual-effect-rows", type=Path, default=V93_CF_EFFECT_ROWS)
    parser.add_argument("--phase5-object-summary", type=Path, default=ROOT / "phase5_object_source_extension/phase5_object_source_extension_summary.json")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "phase6_object_source_counterfactual")
    args = parser.parse_args()

    phase5_rows = read_csv_rows(args.root / "phase5_semantic_carrier_alignment/semantic_carrier_alignment_rows.csv")
    phase5_by_pair = {row["pair_id"]: row for row in phase5_rows}
    object_rows = read_csv_rows(args.object_identity_rows)
    object_by_pair = {row["pair_id"]: row for row in object_rows}
    cf_rows = read_csv_rows(args.v93_counterfactual_effect_rows)
    phase5_object_summary = read_json(args.phase5_object_summary)
    selected_policy = (phase5_object_summary.get("selected_policy") or {}).get("policy")
    if not selected_policy:
        raise ValueError(f"Missing selected policy in {args.phase5_object_summary}")

    joined: list[dict[str, Any]] = []
    missing_phase5_pairs: list[str] = []
    missing_object_pairs: list[str] = []
    for cf_row in cf_rows:
        pair_id = str(cf_row.get("pair_id"))
        row: dict[str, Any] = dict(cf_row)
        phase5 = phase5_by_pair.get(pair_id)
        obj = object_by_pair.get(pair_id)
        if phase5 is None:
            missing_phase5_pairs.append(pair_id)
        else:
            for key, value in phase5.items():
                row.setdefault(key, value)
        if obj is None:
            missing_object_pairs.append(pair_id)
        else:
            for key, value in obj.items():
                if key not in row:
                    row[key] = value
                else:
                    row[f"object_source_{key}"] = value
        joined.append(row)

    thresholds = {
        "boundary_global_cross_ratio_q75": quantile([f(row.get("boundary_global_cross_ratio")) for row in object_rows], 0.75),
        "boundary_new_id_ratio_q75": quantile([f(row.get("boundary_new_id_ratio")) for row in object_rows], 0.75),
        "object_boundary_ratio_q75": quantile([f(row.get("object_boundary_ratio")) for row in object_rows], 0.75),
        "radio_boundary_mean_q75": quantile([f(row.get("radio_boundary_mean")) for row in object_rows], 0.75),
    }
    atoms = build_atoms(joined, thresholds)
    actual_mask = apply_policy(selected_policy, atoms)
    masks = {
        selected_policy: actual_mask,
        "CONTROL_same_count_random": same_count_random(joined, actual_mask),
        "CONTROL_object_order_rot5": object_order_rotate(joined, actual_mask, 5),
        "CONTROL_regime_rot1": regime_rotate(joined, actual_mask, 1),
        "CONTROL_regime_rot2": regime_rotate(joined, actual_mask, 2),
        "CONTROL_same_count_rot7": rotate(actual_mask, 7),
        "V93_CF6_FULL_OBJECT_POLICY": atoms["V93_CF6_FULL_OBJECT_POLICY"],
        "V93_P5_POSITIVE_OBJECT_POLICY": atoms["V93_P5_POSITIVE_OBJECT_POLICY"],
    }
    metric_rows = [counterfactual_metrics(joined, mask, family) for family, mask in masks.items()]
    by_family = {row["family"]: row for row in metric_rows}
    control_families = [name for name in masks if name.startswith("CONTROL_")]
    control_best_bad_ratio = max(
        [by_family[name]["bad_median_residual_improvement_ratio"] for name in control_families if by_family[name]["bad_median_residual_improvement_ratio"] is not None],
        default=None,
    )
    actual = by_family[selected_policy]
    actual_bad_ratio = actual.get("bad_median_residual_improvement_ratio")
    actual_minus_best_control = (
        float(actual_bad_ratio - control_best_bad_ratio)
        if actual_bad_ratio is not None and control_best_bad_ratio is not None
        else None
    )
    for row in metric_rows:
        bad_ratio = row.get("bad_median_residual_improvement_ratio")
        good_ratio = row.get("good_median_residual_worsen_ratio")
        good_max_ratio = row.get("good_max_residual_worsen_ratio")
        row["bad_residual_improvement_gate"] = bad_ratio is not None and bad_ratio >= 0.10
        row["good_median_protection_gate"] = good_ratio is not None and good_ratio <= 0.02
        row["good_catastrophic_worsen_absent"] = good_max_ratio is not None and good_max_ratio <= 0.20
        row["sequence_coverage_gate"] = int(row.get("sequence_coverage") or 0) >= 3
        row["actual_minus_best_control"] = actual_minus_best_control if row["family"] == selected_policy else ""
        row["beats_best_control_gate"] = (
            row["family"] == selected_policy
            and actual_minus_best_control is not None
            and actual_minus_best_control >= 0.05
        )
        row["counterfactual_gate_pass"] = bool(
            row["family"] == selected_policy
            and row["bad_residual_improvement_gate"]
            and row["good_median_protection_gate"]
            and row["good_catastrophic_worsen_absent"]
            and row["beats_best_control_gate"]
            and row["sequence_coverage_gate"]
        )

    selected_effect_rows = effect_rows(joined, actual_mask, selected_policy)
    sign_counts: dict[str, int] = {}
    for row in selected_effect_rows:
        if not bool(row["policy_action"]):
            continue
        key = f"{row.get('base_case_type')}::{row.get('residual_effect_sign')}"
        sign_counts[key] = sign_counts.get(key, 0) + 1

    blocker_parts: list[str] = []
    if not actual.get("bad_residual_improvement_gate"):
        blocker_parts.append("bad_residual_improvement_below_gate")
    if not actual.get("good_median_protection_gate") or not actual.get("good_catastrophic_worsen_absent"):
        blocker_parts.append("good_residual_worsen_exceeds_gate")
    if not actual.get("beats_best_control_gate"):
        blocker_parts.append("does_not_beat_controls")
    if not actual.get("sequence_coverage_gate"):
        blocker_parts.append("sequence_coverage_below_gate")

    pass_rows = [row for row in metric_rows if row["counterfactual_gate_pass"]]
    summary = {
        "phase": "Phase6_object_source_counterfactual",
        "entered": bool(phase5_object_summary.get("object_source_extension_gate_pass")),
        "diagnostic_only": True,
        "phase5_object_source_extension_gate_pass": phase5_object_summary.get("object_source_extension_gate_pass"),
        "selected_policy": selected_policy,
        "phase6_object_source_counterfactual_gate_pass": bool(pass_rows),
        "counterfactual_executed": True,
        "actual_runtime_trajectory_counterfactual_available": False,
        "trace_level_upper_bound_only": True,
        "counterfactual_model": "selected rows cancel observed native merge/gauge trace deltas; no trajectory rerun is claimed",
        "actual_family": actual,
        "control_best_bad_median_residual_improvement_ratio": control_best_bad_ratio,
        "actual_minus_best_control": actual_minus_best_control,
        "selected_action_residual_effect_sign_counts": sign_counts,
        "passing_families": [row["family"] for row in pass_rows],
        "blocker": "" if pass_rows else ";".join(blocker_parts),
        "runtime_action_allowed": bool(pass_rows),
        "ttt_allowed": False,
        "joined_rows": len(joined),
        "missing_phase5_pairs": missing_phase5_pairs,
        "missing_object_source_pairs": missing_object_pairs,
        "thresholds": thresholds,
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "counterfactual_policy_metrics.csv", metric_rows)
    write_csv(args.out_dir / "counterfactual_effect_rows.csv", selected_effect_rows)
    write_json(args.out_dir / "phase6_object_source_counterfactual_summary.json", summary)
    (args.out_dir / "analysis.md").write_text(
        "\n".join(
            [
                "# ACL2 v94 Phase6 Object-Source Counterfactual",
                "",
                f"- selected_policy: `{selected_policy}`",
                f"- phase6_object_source_counterfactual_gate_pass: `{summary['phase6_object_source_counterfactual_gate_pass']}`",
                f"- blocker: `{summary['blocker']}`",
                f"- bad_median_residual_improvement_ratio: `{actual.get('bad_median_residual_improvement_ratio')}`",
                f"- good_max_residual_worsen_ratio: `{actual.get('good_max_residual_worsen_ratio')}`",
                f"- actual_minus_best_control: `{actual_minus_best_control}`",
                f"- selected_action_residual_effect_sign_counts: `{sign_counts}`",
                "",
                "This is a trace-level upper-bound diagnostic only; no runtime trajectory counterfactual or TTT run is claimed.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"phase6_object_source_counterfactual_gate_pass={summary['phase6_object_source_counterfactual_gate_pass']}")
    print(f"selected_policy={selected_policy}")
    print(f"blocker={summary['blocker']}")
    print(f"bad_median_residual_improvement_ratio={actual.get('bad_median_residual_improvement_ratio')}")
    print(f"good_median_residual_worsen_ratio={actual.get('good_median_residual_worsen_ratio')}")
    print(f"good_max_residual_worsen_ratio={actual.get('good_max_residual_worsen_ratio')}")
    print(f"actual_minus_best_control={actual_minus_best_control}")
    print(f"runtime_action_allowed={summary['runtime_action_allowed']}")


if __name__ == "__main__":
    main()
