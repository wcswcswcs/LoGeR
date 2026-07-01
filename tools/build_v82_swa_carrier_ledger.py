#!/usr/bin/env python3
"""Build v82 SWA carrier ledger from Phase3 runtime route manifest."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


DEFAULT_ROOT = Path(
    "results/acl2_v82tf_swa_carrier_semantic_scale_handoff/"
    "phase4_swa_carrier_ledger"
)
DEFAULT_VISUAL_MANIFEST = Path(
    "results/acl2_v82tf_swa_carrier_semantic_scale_handoff/"
    "phase3_swa_true_route_visual_confirmation/visual_manifest.csv"
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _corr(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 3 or len(xs) != len(ys):
        return None
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 0.0 or vy <= 0.0:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / math.sqrt(vx * vy)


def _route_entropy(mean: float | None, nonzero_ratio: float | None) -> float | None:
    if mean is None:
        return None
    p = min(1.0, max(0.0, mean))
    if p in {0.0, 1.0}:
        entropy = 0.0
    else:
        entropy = -(p * math.log(p) + (1.0 - p) * math.log(1.0 - p))
    if nonzero_ratio is not None:
        entropy *= min(1.0, max(0.0, nonzero_ratio))
    return entropy


def _ledger_row(row: dict[str, str], family: str) -> dict[str, Any]:
    if family == "V-protect":
        prefix = "source_replace"
        side = "cache"
        tap = "V"
        action_type = "V stable protect"
    elif family == "K-risk":
        prefix = "source_gate"
        side = "cache"
        tap = "K"
        action_type = "K risk veto"
    elif family == "Q-conditioned":
        prefix = "source_gate"
        side = "current"
        tap = "Q"
        action_type = "Q-conditioned pair bias"
    else:
        prefix = "source_replace"
        side = "cache"
        tap = "V"
        action_type = "context floor"

    score_mean = _float(row.get(f"{prefix}_score_mean"))
    control_mean = _float(row.get(f"{prefix}_control_mean"))
    dq_mean = _float(row.get(f"{prefix}_Dq_mean"))
    ds_mean = _float(row.get(f"{prefix}_Ds_mean"))
    nonzero = _float(row.get(f"{prefix}_score_nonzero_ratio"))
    stable = _float(row.get("stable_overlap_mass"))
    harm = _float(row.get("harm_overlap_mass"))
    context = _float(row.get("context_overlap_mass"))
    selected_minus_control = (
        score_mean - control_mean if score_mean is not None and control_mean is not None else None
    )
    stable_alignment_delta = (
        stable * selected_minus_control if stable is not None and selected_minus_control is not None else None
    )
    harm_alignment_delta = (
        harm * selected_minus_control if harm is not None and selected_minus_control is not None else None
    )
    qk_delta = dq_mean - ds_mean if dq_mean is not None and ds_mean is not None else None
    return {
        "seq": row.get("seq", ""),
        "prev_chunk": row.get("prev_chunk", ""),
        "curr_chunk": row.get("curr_chunk", ""),
        "case_type": row.get("case_type", ""),
        "base_case_type": row.get("base_case_type", ""),
        "quality_type": row.get("quality_type", ""),
        "carrier_family": family,
        "side": side,
        "tap": tap,
        "layer": row.get(f"{prefix}_swa_layer_idx", ""),
        "head": "all_head_aggregate",
        "action_type": action_type,
        "actual_route_mass": score_mean,
        "control_route_mass": control_mean,
        "random_route_mass": "",
        "same_head_random_available": False,
        "shuffled_semantic_available": False,
        "per_head_available": False,
        "actual_vs_random_l1": row.get("actual_vs_random_l1", ""),
        "stable_alignment_delta": stable_alignment_delta,
        "harm_alignment_delta": harm_alignment_delta,
        "V_selected_minus_random": selected_minus_control if tap == "V" else "",
        "K_selected_minus_random": selected_minus_control if tap == "K" else "",
        "QK_compatibility_delta": qk_delta,
        "route_entropy": _route_entropy(score_mean, nonzero),
        "stable_overlap_mass": stable,
        "harm_overlap_mass": harm,
        "context_overlap_mass": context,
        "future_after_overlap": "",
        "boundary_jump": "",
        "overlap_scale_residual": "",
        "head_layer_sensitivity_score": "",
        "head_layer_sensitivity_status": "not_localized_all_head_aggregate_only",
        "good_case_false_positive": bool(
            row.get("base_case_type") == "good"
            and selected_minus_control is not None
            and selected_minus_control > 0.0
        ),
        "carrier_visual_evidence_confirmed": _truthy(row.get("has_actual_route_mask"))
        and _truthy(row.get("has_qkv_maps"))
        and _truthy(row.get("actual_vs_random_difference_reviewed")),
        "visual_file": row.get("visual_file", ""),
        "route_file": row.get(f"{prefix}_route_file", ""),
    }


def _merge_pair_metrics(rows: list[dict[str, Any]], source_rows: list[dict[str, str]]) -> None:
    by_key = {
        (r.get("seq", ""), str(r.get("prev_chunk", "")), str(r.get("curr_chunk", ""))): r
        for r in source_rows
    }
    for row in rows:
        src = by_key.get((row["seq"], str(row["prev_chunk"]), str(row["curr_chunk"])), {})
        for key in ["future_after_overlap", "boundary_jump", "overlap_scale_residual"]:
            row[key] = src.get(key, "")


def _summaries(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_family[row["carrier_family"]].append(row)
    for family, family_rows in sorted(by_family.items()):
        scores: list[float] = []
        overlap_scale: list[float] = []
        future: list[float] = []
        boundary: list[float] = []
        bad_signs: list[float] = []
        good_false_positive = 0
        for row in family_rows:
            score = _float(row.get("actual_route_mass"))
            if score is not None:
                for metric_name, metric_values in [
                    ("overlap_scale_residual", overlap_scale),
                    ("future_after_overlap", future),
                    ("boundary_jump", boundary),
                ]:
                    metric = _float(row.get(metric_name))
                    if metric is not None:
                        if metric_values is overlap_scale:
                            scores.append(score)
                        metric_values.append(metric)
                if row.get("base_case_type") == "bad":
                    delta = _float(row.get("stable_alignment_delta"))
                    if delta is not None:
                        bad_signs.append(delta)
            if row.get("good_case_false_positive") is True:
                good_false_positive += 1
        # Keep correlations separate because some metric lists can differ after missing filtering.
        corr_inputs: dict[str, tuple[list[float], list[float]]] = {
            "overlap_scale_residual_correlation": ([], []),
            "future_after_overlap_correlation": ([], []),
            "boundary_jump_correlation": ([], []),
        }
        for row in family_rows:
            score = _float(row.get("actual_route_mass"))
            if score is None:
                continue
            for out_key, metric_name in [
                ("overlap_scale_residual_correlation", "overlap_scale_residual"),
                ("future_after_overlap_correlation", "future_after_overlap"),
                ("boundary_jump_correlation", "boundary_jump"),
            ]:
                metric = _float(row.get(metric_name))
                if metric is not None:
                    corr_inputs[out_key][0].append(score)
                    corr_inputs[out_key][1].append(metric)
        summary[family] = {
            "rows": len(family_rows),
            "seq_coverage": sorted({row.get("seq", "") for row in family_rows if row.get("seq", "")}),
            "visual_confirmed_rows": sum(1 for row in family_rows if row.get("carrier_visual_evidence_confirmed")),
            "bad_stable_alignment_positive_count": sum(1 for value in bad_signs if value > 0.0),
            "bad_stable_alignment_count": len(bad_signs),
            "good_case_false_positive_count": good_false_positive,
            "per_head_available": False,
            "same_head_random_available": False,
            "shuffled_semantic_available": False,
            **{
                out_key: _corr(xs, ys)
                for out_key, (xs, ys) in corr_inputs.items()
            },
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--visual-manifest", type=Path, default=DEFAULT_VISUAL_MANIFEST)
    parser.add_argument("--pair-bank", type=Path, default=Path(
        "results/acl2_v82tf_swa_carrier_semantic_scale_handoff/"
        "phase2_swa_pair_bank_v2/swa_pair_bank_v2.csv"
    ))
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()

    manifest = _read_csv(args.visual_manifest)
    pair_bank = _read_csv(args.pair_bank)
    rows: list[dict[str, Any]] = []
    for row in manifest:
        for family in ["V-protect", "K-risk", "Q-conditioned", "context-floor"]:
            rows.append(_ledger_row(row, family))
    _merge_pair_metrics(rows, pair_bank)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.out_dir / "swa_carrier_ledger.csv", rows)
    summary = {
        "schema": "acl2_v82_swa_carrier_ledger_v1",
        "rows": len(rows),
        "visual_rows": len(manifest),
        "families": _summaries(rows),
        "limitations": [
            "runtime route tensors are layer aggregate and do not expose per-head routing",
            "same-head random and shuffled semantic controls are not present in Phase3 dumps",
            "READ selected stable/harm masks are not token-aligned in the current artifacts",
        ],
    }
    (args.out_dir / "swa_carrier_ledger_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"rows": len(rows), "out_dir": str(args.out_dir)}, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
