#!/usr/bin/env python3
"""Rebuild ACL2 v96 Track E/C SWA route decision artifacts.

This is diagnostic-only. It joins the Track E raw SWA transport case rows with
the Track A case atlas, evaluates simple route separability checks, and writes
the Track C latent-gauge summary plus a route decision JSON/Markdown.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import median
from typing import Any


ROOT = Path("results/acl2_v96tf_vggt4d_informed_semantic_gauge_preserving_memory_control")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=False) + "\n")


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _write_rows(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


def _f(row: dict[str, Any], key: str, default: float = math.nan) -> float:
    value = row.get(key, "")
    if value in ("", None):
        return default
    return float(value)


def _pearson(xs: list[float], ys: list[float]) -> float:
    pairs = [(x, y) for x, y in zip(xs, ys) if math.isfinite(x) and math.isfinite(y)]
    if len(pairs) < 2:
        return math.nan
    xvals, yvals = zip(*pairs)
    mx = sum(xvals) / len(xvals)
    my = sum(yvals) / len(yvals)
    vx = sum((x - mx) ** 2 for x in xvals)
    vy = sum((y - my) ** 2 for y in yvals)
    if vx <= 0 or vy <= 0:
        return math.nan
    cov = sum((x - mx) * (y - my) for x, y in pairs)
    return cov / math.sqrt(vx * vy)


def _ranks(values: list[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [math.nan] * len(values)
    i = 0
    while i < len(indexed):
        j = i + 1
        while j < len(indexed) and indexed[j][1] == indexed[i][1]:
            j += 1
        rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[indexed[k][0]] = rank
        i = j
    return ranks


def _spearman(xs: list[float], ys: list[float]) -> float:
    pairs = [(x, y) for x, y in zip(xs, ys) if math.isfinite(x) and math.isfinite(y)]
    if len(pairs) < 2:
        return math.nan
    xvals, yvals = zip(*pairs)
    return _pearson(_ranks(list(xvals)), _ranks(list(yvals)))


def _best_threshold(values: list[float], labels: list[int], direction: str) -> dict[str, Any]:
    pos = sum(1 for label in labels if label == 1)
    neg = sum(1 for label in labels if label == 0)
    best: dict[str, Any] | None = None
    for threshold in sorted(set(values)):
        if direction == "lower_bad":
            preds = [1 if value <= threshold else 0 for value in values]
        elif direction == "higher_bad":
            preds = [1 if value >= threshold else 0 for value in values]
        else:
            raise ValueError(f"unsupported direction: {direction}")
        tp = sum(1 for pred, label in zip(preds, labels) if pred == 1 and label == 1)
        tn = sum(1 for pred, label in zip(preds, labels) if pred == 0 and label == 0)
        tpr = tp / pos if pos else 0.0
        tnr = tn / neg if neg else 0.0
        score = 0.5 * (tpr + tnr)
        candidate = {
            "balanced_accuracy": score,
            "threshold": threshold,
            "direction": direction,
            "tp": tp,
            "tn": tn,
            "pos": pos,
            "neg": neg,
        }
        if best is None or (score, tp + tn) > (best["balanced_accuracy"], best["tp"] + best["tn"]):
            best = candidate
    return best or {
        "balanced_accuracy": 0.0,
        "threshold": math.nan,
        "direction": direction,
        "tp": 0,
        "tn": 0,
        "pos": pos,
        "neg": neg,
    }


def _clean_float(value: float) -> float | None:
    return None if not math.isfinite(value) else value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument(
        "--tracke-root",
        type=Path,
        default=ROOT / "trackE_swa_raw_transport_trace_swa_atlas_v1",
    )
    parser.add_argument("--tracka-rows", type=Path, default=ROOT / "trackA_case_response_atlas" / "rows.csv")
    parser.add_argument("--trackc-out", type=Path, default=ROOT / "trackC_latent_gauge_alignment")
    parser.add_argument("--route-out", type=Path, default=ROOT / "route_decisions")
    args = parser.parse_args()

    tracke_summary_path = args.tracke_root / "trackE_swa_raw_transport_trace_summary.json"
    tracke_rows_path = args.tracke_root / "trackE_swa_raw_transport_trace_case_rows.csv"
    tracke_summary = _read_json(tracke_summary_path)
    tracke_rows = _read_rows(tracke_rows_path)
    tracka_by_case = {row["case_id"]: row for row in _read_rows(args.tracka_rows)}

    joined: list[dict[str, Any]] = []
    for row in tracke_rows:
        atlas = tracka_by_case[row["case_id"]]
        bucket = row["bucket"]
        bad_label = 0 if bucket == "SWA_HANDOFF_GOOD_CONTROL" else 1
        joined.append(
            {
                "case_id": row["case_id"],
                "seq": row["seq"],
                "prev_chunk": row["prev_chunk"],
                "curr_chunk": row["curr_chunk"],
                "bucket": bucket,
                "bad_label": bad_label,
                "L3_handoff_transfer_penalty_proxy": _f(atlas, "L3_handoff_transfer_penalty_proxy"),
                "L3_J_handoff": _f(atlas, "L3_J_handoff"),
                "L3_adjacent_log_scale_jump": _f(atlas, "L3_adjacent_log_scale_jump"),
                "stable_pair_mass": _f(row, "trace_swa_raw_transport_stable_pair_mass_mean"),
                "unreliable_pair_mass": _f(row, "trace_swa_raw_transport_unreliable_pair_mass_mean"),
                "stable_actual_minus_random": _f(row, "trace_swa_raw_transport_stable_actual_minus_random_mean"),
                "unreliable_actual_minus_random": _f(row, "trace_swa_raw_transport_unreliable_actual_minus_random_mean"),
                "qk_similarity_mean": _f(row, "trace_swa_raw_transport_qk_similarity_mean"),
                "qk_similarity_max_mean": _f(row, "trace_swa_raw_transport_qk_similarity_max_mean"),
                "route_entropy": _f(row, "trace_swa_raw_transport_route_entropy_mean"),
                "feature_residual": _f(row, "trace_swa_raw_transport_feature_residual_mean"),
                "cache_k_stability": _f(row, "trace_swa_raw_transport_cache_k_stability_mean"),
                "cache_v_stability": _f(row, "trace_swa_raw_transport_cache_v_stability_mean"),
                "trace_available": _f(row, "trace_swa_raw_transport_available_frac"),
                "stable_nonempty": _f(row, "trace_swa_raw_transport_stable_nonempty_frac"),
                "unreliable_nonempty": _f(row, "trace_swa_raw_transport_unreliable_nonempty_frac"),
            }
        )

    labels = [int(row["bad_label"]) for row in joined]
    l3 = [_f(row, "L3_handoff_transfer_penalty_proxy") for row in joined]
    metric_specs = [
        ("stable_pair_mass_lower_bad", "stable_pair_mass", "lower_bad"),
        ("unreliable_pair_mass_higher_bad", "unreliable_pair_mass", "higher_bad"),
        ("feature_residual_higher_bad", "feature_residual", "higher_bad"),
        ("route_entropy_higher_bad", "route_entropy", "higher_bad"),
        ("qk_similarity_mean_lower_bad", "qk_similarity_mean", "lower_bad"),
        ("cache_v_stability_lower_bad", "cache_v_stability", "lower_bad"),
    ]
    metric_summary: dict[str, Any] = {}
    for name, key, direction in metric_specs:
        values = [_f(row, key) for row in joined]
        metric_summary[name] = {
            "key": key,
            "best_threshold_ba": _best_threshold(values, labels, direction),
            "pearson_with_L3_handoff_transfer_penalty_proxy": _clean_float(_pearson(values, l3)),
            "spearman_with_L3_handoff_transfer_penalty_proxy": _clean_float(_spearman(values, l3)),
        }

    max_ba = max(item["best_threshold_ba"]["balanced_accuracy"] for item in metric_summary.values())
    max_abs_pearson = max(abs(item["pearson_with_L3_handoff_transfer_penalty_proxy"] or 0.0) for item in metric_summary.values())
    stable_random = [row["stable_actual_minus_random"] for row in joined]
    unreliable_random = [row["unreliable_actual_minus_random"] for row in joined]
    random_margin_pass = abs(median(stable_random)) >= 0.05 and abs(median(unreliable_random)) >= 0.05
    gate_pass = bool(max_ba >= 0.70 and max_abs_pearson >= 0.30 and random_margin_pass)
    trackc_summary = {
        "schema": "acl2_v96_trackc_swa_latent_gauge_alignment_from_raw_trace_v1",
        "status": "complete",
        "track": "trackC_latent_gauge_alignment",
        "source_trace_summary": str(tracke_summary_path),
        "case_count": len(joined),
        "non_good_count": sum(labels),
        "good_control_count": len(labels) - sum(labels),
        "metric_summary": metric_summary,
        "max_balanced_accuracy": max_ba,
        "max_abs_pearson_with_L3": max_abs_pearson,
        "median_stable_actual_minus_random": median(stable_random),
        "median_unreliable_actual_minus_random": median(unreliable_random),
        "gate_rule": "stable alignment gap separates bad/good by BA >= 0.70; beats random/shuffle margins >= 0.05; correlates with L3 handoff_transfer_penalty",
        "gate_pass": gate_pass,
        "method_success": False,
        "runtime_action_allowed": False,
        "full_method_success": False,
        "classification": "TRACE_PASS_ACTION_FAIL_ROUTE_NOT_HANDOFF_CARRIER" if not gate_pass else "TRACE_PASS_ACTION_ELIGIBLE_DIAGNOSTIC_ONLY",
        "failure_reason": "Raw SWA trace exists, but latent route/alignment metrics do not satisfy Track C separation/random-margin/correlation gate."
        if not gate_pass
        else "",
    }

    row_fields = list(joined[0].keys()) if joined else []
    _write_rows(args.trackc_out / "rows.csv", joined, row_fields)
    _write_json(args.trackc_out / "summary.json", trackc_summary)
    _write_rows(
        args.trackc_out / "gate_checks.csv",
        [
            {"check": "max_balanced_accuracy_ge_0p70", "value": max_ba, "pass": max_ba >= 0.70},
            {"check": "max_abs_pearson_ge_0p30", "value": max_abs_pearson, "pass": max_abs_pearson >= 0.30},
            {"check": "median_random_margin_abs_ge_0p05", "value": min(abs(median(stable_random)), abs(median(unreliable_random))), "pass": random_margin_pass},
            {"check": "gate_pass", "value": gate_pass, "pass": gate_pass},
        ],
        ["check", "value", "pass"],
    )
    (args.trackc_out / "failure_report.md").write_text(
        "# Track C SWA Latent Gauge Alignment\n\n"
        f"classification = {trackc_summary['classification']}\n\n"
        f"gate_pass = {gate_pass}\n\n"
        f"max_balanced_accuracy = {max_ba}\n\n"
        f"max_abs_pearson_with_L3 = {max_abs_pearson}\n\n"
        f"median_stable_actual_minus_random = {median(stable_random)}\n\n"
        f"median_unreliable_actual_minus_random = {median(unreliable_random)}\n",
    )
    (args.trackc_out / "what_would_have_to_be_true_to_pass.md").write_text(
        "# What Would Have To Be True To Pass\n\n"
        "- A route metric must separate SWA non-good handoff cases from good controls with BA >= 0.70.\n"
        "- Stable/unreliable actual-minus-random medians must clear the 0.05 margin.\n"
        "- The route metric must correlate with L3 handoff transfer penalty, not just boundary/proxy behavior.\n",
    )

    gates = tracke_summary.get("gates", {})
    separability = tracke_summary.get("separability", {})
    route_mass_sep = bool(gates.get("bad_good_separable_by_stable_or_unreliable_mass_margin_ge_0p05", False))
    route_decision = {
        "schema": "acl2_v96_route_decision_v1",
        "decision_id": "trackE_trackC_swa_route_not_handoff_carrier_20260628",
        "source_artifacts": {
            "trackE_raw_trace_summary": str(tracke_summary_path),
            "trackC_latent_gauge_summary": str(args.trackc_out / "summary.json"),
        },
        "swa_raw_trace_available": bool(gates.get("trace_availability_ge_0p90", False)),
        "swa_raw_trace_availability_frac": gates.get("trace_availability_frac"),
        "stable_group_nonempty_frac": gates.get("stable_group_nonempty_frac"),
        "unreliable_group_nonempty_frac": gates.get("unreliable_group_nonempty_frac"),
        "per_layer_per_head_rows_available": bool(tracke_summary.get("per_layer_per_head_rows_available", False)),
        "bad_good_separable_by_route_mass_margin_ge_0p05": route_mass_sep,
        "route_mass_stable_bad_lower_margin_median": separability.get("stable_mass_bad_lower_margin_median"),
        "route_mass_unreliable_bad_higher_margin_median": separability.get("unreliable_mass_bad_higher_margin_median"),
        "trackC_gate_pass": gate_pass,
        "trackC_max_balanced_accuracy": max_ba,
        "trackC_max_abs_pearson_with_L3": max_abs_pearson,
        "trackC_median_stable_actual_minus_random": median(stable_random),
        "trackC_median_unreliable_actual_minus_random": median(unreliable_random),
        "classification": "SWA_ROUTE_NOT_HANDOFF_CARRIER_DIAGNOSTIC_ONLY"
        if not route_mass_sep or not gate_pass
        else "SWA_ROUTE_CARRIER_DIAGNOSTIC_ONLY",
        "runtime_swa_action_allowed": False,
        "runtime_ttt_action_allowed": False,
        "method_success": False,
        "full_method_success": False,
        "next_route_recommendation": "Do not run SWA action. Return to READ global tradeoff analysis and TTT write trace instrumentation; Track F remains blocked until persistent write traces exist.",
    }
    _write_json(args.route_out / "trackE_trackC_swa_route_decision.json", route_decision)
    (args.route_out / "trackE_trackC_swa_route_decision.md").write_text(
        "# Track E/C SWA Route Decision\n\n"
        f"classification = {route_decision['classification']}\n\n"
        f"runtime_swa_action_allowed = {route_decision['runtime_swa_action_allowed']}\n\n"
        f"trackC_gate_pass = {gate_pass}\n\n"
        f"route_mass_stable_bad_lower_margin_median = {route_decision['route_mass_stable_bad_lower_margin_median']}\n\n"
        f"route_mass_unreliable_bad_higher_margin_median = {route_decision['route_mass_unreliable_bad_higher_margin_median']}\n",
    )

    print(json.dumps(route_decision, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
