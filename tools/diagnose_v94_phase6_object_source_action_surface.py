#!/usr/bin/env python3
"""Evaluate measured runtime-probe action surface for v94 object-source rows."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path("results/acl2_v94tf_semantic_gauge_failure_localization_causal_memory_control")
DEFAULT_PROBE_DIRS = [
    ROOT / "phase3s_merge_gauge_actuator_sweep_max16_confirm",
    ROOT / "phase5_semantic_role_carrier_probe",
    ROOT / "phase5_remaining_labelled_carrier_probe",
]


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


def median(values: list[float]) -> float | None:
    finite_values = sorted(value for value in values if math.isfinite(value))
    if not finite_values:
        return None
    mid = len(finite_values) // 2
    if len(finite_values) % 2:
        return float(finite_values[mid])
    return float((finite_values[mid - 1] + finite_values[mid]) / 2.0)


def max_or_none(values: list[float]) -> float | None:
    finite_values = [value for value in values if math.isfinite(value)]
    if not finite_values:
        return None
    return float(max(finite_values))


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
        ordered = sorted(indices, key=lambda idx: stable_key("v94_action_surface_same_count", seq, label, rows[idx].get("pair_id")))
        for idx in ordered[: min(count, len(ordered))]:
            out[idx] = True
    return out


def seqs_for(rows: list[dict[str, Any]], mask: list[bool]) -> set[str]:
    return {str(row.get("seq")) for row, selected in zip(rows, mask) if selected}


def metric(rows: list[dict[str, Any]], mask: list[bool], name: str) -> dict[str, Any]:
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
    bad_median = median(bad_i)
    good_median = median(good_w)
    good_max = max_or_none(good_w)
    return {
        "selection": name,
        "selected_row_count": len(selected),
        "bad_rows": len(bad),
        "good_rows": len(good),
        "sequence_coverage": len(seqs_for(rows, mask)),
        "bad_median_I_J_runtime_proxy": bad_median,
        "bad_min_I_J_runtime_proxy": min(bad_i) if bad_i else None,
        "bad_negative_improvement_rows": sum(1 for value in bad_i if math.isfinite(value) and value < 0),
        "good_median_worsen_runtime_proxy": good_median,
        "good_max_worsen_runtime_proxy": good_max,
        "good_worsen_gt_0p02_rows": sum(1 for value in good_w if math.isfinite(value) and value > 0.02),
        "carrier_state_delta_nonzero_rows": sum(1 for value in carrier_delta if math.isfinite(value) and value > 1e-9),
        "trajectory_rows_complete": all(trajectory_available) if selected else False,
        "bad_hits": ",".join(str(rows[idx].get("pair_id")) for idx in bad),
        "good_hits": ",".join(str(rows[idx].get("pair_id")) for idx in good),
    }


def load_effect_rows(probe_dirs: list[Path], variant: str) -> list[dict[str, Any]]:
    by_pair: dict[str, dict[str, Any]] = {}
    source_order: dict[str, int] = {}
    for order, probe_dir in enumerate(probe_dirs):
        path = probe_dir / "runtime_probe_effect_rows.csv"
        if not path.exists():
            continue
        for row in read_csv_rows(path):
            if row.get("variant") != variant:
                continue
            if row.get("case_label_offline_only") not in {"bad", "good"}:
                continue
            pair_id = str(row.get("pair_id"))
            if pair_id not in by_pair or order < source_order[pair_id]:
                out = dict(row)
                out["source_probe_dir"] = str(probe_dir)
                by_pair[pair_id] = out
                source_order[pair_id] = order
    return [by_pair[pair_id] for pair_id in sorted(by_pair)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--probe-dirs", default=",".join(str(path) for path in DEFAULT_PROBE_DIRS))
    parser.add_argument("--variant", default="merge_alpha_0p2")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "phase6_object_source_action_surface")
    args = parser.parse_args()

    probe_dirs = [Path(item) for item in args.probe_dirs.split(",") if item.strip()]
    selected_rows = read_csv_rows(args.root / "phase5_object_source_extension/selected_policy_rows.csv")
    object_selected = {
        row["pair_id"]
        for row in selected_rows
        if row.get("selected_policy_positive") == "True"
        and row.get("case_label_offline_only") in {"bad", "good"}
    }
    selected_meta = {row["pair_id"]: row for row in selected_rows}
    effect_rows = load_effect_rows(probe_dirs, args.variant)
    for row in effect_rows:
        meta = selected_meta.get(str(row.get("pair_id")), {})
        row["object_source_policy_positive"] = str(row.get("pair_id")) in object_selected
        row["semantic_evidence_type"] = meta.get("semantic_evidence_type", "")
        row["boundary_global_cross_ratio"] = meta.get("boundary_global_cross_ratio", "")
        row["object_boundary_ratio"] = meta.get("object_boundary_ratio", "")

    measured_pairs = {str(row.get("pair_id")) for row in effect_rows}
    missing_selected_pairs = sorted(object_selected - measured_pairs)
    actual_mask = [bool(row.get("object_source_policy_positive")) for row in effect_rows]
    control_masks = {
        "CONTROL_same_count_random": same_count_random(effect_rows, actual_mask),
        "CONTROL_object_order_rot5": object_order_rotate(effect_rows, actual_mask, 5),
        "CONTROL_regime_rot1": regime_rotate(effect_rows, actual_mask, 1),
        "CONTROL_regime_rot2": regime_rotate(effect_rows, actual_mask, 2),
        "CONTROL_same_count_rot7": rotate(actual_mask, 7),
    }
    actual = metric(effect_rows, actual_mask, "OBJECT_SOURCE_POLICY")
    control_rows = [metric(effect_rows, mask, name) for name, mask in control_masks.items()]
    best_control_bad = max(
        [row["bad_median_I_J_runtime_proxy"] for row in control_rows if row["bad_median_I_J_runtime_proxy"] is not None],
        default=None,
    )
    actual_bad = actual.get("bad_median_I_J_runtime_proxy")
    actual_minus_best_control = (
        float(actual_bad - best_control_bad) if actual_bad is not None and best_control_bad is not None else None
    )
    bad_gate = actual_bad is not None and actual_bad >= 0.05
    good_median = actual.get("good_median_worsen_runtime_proxy")
    good_max = actual.get("good_max_worsen_runtime_proxy")
    good_median_gate = good_median is not None and good_median <= 0.02
    good_max_gate = good_max is not None and good_max <= 0.05
    sequence_gate = int(actual.get("sequence_coverage") or 0) >= 3
    trajectory_gate = bool(actual.get("trajectory_rows_complete"))
    carrier_delta_gate = int(actual.get("carrier_state_delta_nonzero_rows") or 0) > 0
    control_gate = actual_minus_best_control is not None and actual_minus_best_control >= 0.05
    coverage_gate = len(missing_selected_pairs) == 0 and len(object_selected) > 0
    gate_pass = bool(
        coverage_gate
        and bad_gate
        and good_median_gate
        and good_max_gate
        and sequence_gate
        and trajectory_gate
        and carrier_delta_gate
        and control_gate
    )
    blocker_parts = []
    if not coverage_gate:
        blocker_parts.append("selected_pair_runtime_probe_coverage_incomplete")
    if not bad_gate:
        blocker_parts.append("bad_runtime_proxy_improvement_below_gate")
    if not good_median_gate or not good_max_gate:
        blocker_parts.append("good_runtime_proxy_worsen_exceeds_gate")
    if not control_gate:
        blocker_parts.append("does_not_beat_measured_selection_controls")
    if not carrier_delta_gate or not trajectory_gate:
        blocker_parts.append("action_surface_unavailable")
    if not sequence_gate:
        blocker_parts.append("sequence_coverage_below_gate")
    summary = {
        "phase": "Phase6_object_source_action_surface",
        "diagnostic_only": True,
        "variant": args.variant,
        "probe_dirs": [str(path) for path in probe_dirs],
        "measured_labelled_pair_count": len(measured_pairs),
        "object_source_selected_pair_count": len(object_selected),
        "missing_selected_pairs": missing_selected_pairs,
        "coverage_gate": coverage_gate,
        "actual_family": actual,
        "control_families": control_rows,
        "best_control_bad_median_I_J_runtime_proxy": best_control_bad,
        "actual_minus_best_control": actual_minus_best_control,
        "bad_runtime_proxy_improvement_gate": bad_gate,
        "good_median_protection_gate": good_median_gate,
        "good_catastrophic_worsen_absent": good_max_gate,
        "sequence_coverage_gate": sequence_gate,
        "trajectory_rows_complete_gate": trajectory_gate,
        "carrier_state_delta_gate": carrier_delta_gate,
        "beats_measured_selection_controls_gate": control_gate,
        "semantic_not_specific": not control_gate,
        "phase6_object_source_action_surface_gate_pass": gate_pass,
        "blocker": "" if gate_pass else ";".join(blocker_parts),
        "runtime_action_allowed": gate_pass,
        "ttt_allowed": False,
        "note": "Measured pipeline replay/action-surface diagnostic over completed runtime probes; not a promoted online policy.",
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "action_surface_effect_rows.csv", effect_rows)
    write_csv(args.out_dir / "action_surface_control_metrics.csv", [actual, *control_rows])
    write_json(args.out_dir / "phase6_object_source_action_surface_summary.json", summary)
    (args.out_dir / "analysis.md").write_text(
        "\n".join(
            [
                "# ACL2 v94 Phase6 Object-Source Action Surface",
                "",
                f"- variant: `{args.variant}`",
                f"- phase6_object_source_action_surface_gate_pass: `{gate_pass}`",
                f"- blocker: `{summary['blocker']}`",
                f"- selected_pair_coverage: `{len(object_selected) - len(missing_selected_pairs)}/{len(object_selected)}`",
                f"- bad_median_I_J_runtime_proxy: `{actual.get('bad_median_I_J_runtime_proxy')}`",
                f"- good_max_worsen_runtime_proxy: `{actual.get('good_max_worsen_runtime_proxy')}`",
                f"- actual_minus_best_control: `{actual_minus_best_control}`",
                f"- bad_negative_improvement_rows: `{actual.get('bad_negative_improvement_rows')}`",
                "",
                "This is real measured runtime-probe evidence from completed two-window probes, not trace zeroing.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"phase6_object_source_action_surface_gate_pass={gate_pass}")
    print(f"blocker={summary['blocker']}")
    print(f"selected_pair_coverage={len(object_selected) - len(missing_selected_pairs)}/{len(object_selected)}")
    print(f"measured_labelled_pair_count={len(measured_pairs)}")
    print(f"bad_median_I_J_runtime_proxy={actual.get('bad_median_I_J_runtime_proxy')}")
    print(f"good_median_worsen_runtime_proxy={actual.get('good_median_worsen_runtime_proxy')}")
    print(f"good_max_worsen_runtime_proxy={actual.get('good_max_worsen_runtime_proxy')}")
    print(f"actual_minus_best_control={actual_minus_best_control}")
    print(f"bad_negative_improvement_rows={actual.get('bad_negative_improvement_rows')}")
    print(f"runtime_action_allowed={summary['runtime_action_allowed']}")


if __name__ == "__main__":
    main()
