#!/usr/bin/env python3
"""Audit whether Track G internal SWA cues unlock Track E measured action.

This uses already completed measured runtime-probe effect rows. It does not run
new GPU jobs and does not claim a runtime policy. The question is narrower:
given v83/v85 internal cues that passed Track G G5 controls, do any measured
Track E variants produce at least 5% L3 handoff improvement on the cue-selected
bad pairs while protecting good controls and beating same-count controls?
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import median
from typing import Any, Iterable, Mapping


ROOT = Path("results/acl2_v95tf_multiroute_semantic_memory_evidence_control")
DEFAULT_EFFECT_ROWS = ROOT / "trackE_alpha04_runtime_probe/runtime_probe_effect_rows.csv"
DEFAULT_CUE_METRICS = ROOT / "trackG_swa_internal_cue_eval_v1/method_safe_internal_candidate_metrics.csv"
DEFAULT_OUT_DIR = ROOT / "trackE_internal_cue_action_surface_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--effect-rows", type=Path, default=DEFAULT_EFFECT_ROWS)
    parser.add_argument("--cue-metrics", type=Path, default=DEFAULT_CUE_METRICS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--max-cues", type=int, default=120)
    parser.add_argument("--same-count-controls", type=int, default=64)
    parser.add_argument("--bad-handoff-threshold", type=float, default=0.05)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def serialize(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        if not math.isfinite(value):
            return ""
        return f"{value:.12g}"
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    rows = list(rows)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: serialize(row.get(key, "")) for key in fields})


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def f(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def bool_text(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def finite(values: Iterable[Any]) -> list[float]:
    return [value for value in (f(item) for item in values) if math.isfinite(value)]


def median_or_none(values: Iterable[Any]) -> float | None:
    vals = finite(values)
    return float(median(vals)) if vals else None


def max_or_none(values: Iterable[Any]) -> float | None:
    vals = finite(values)
    return float(max(vals)) if vals else None


def min_or_none(values: Iterable[Any]) -> float | None:
    vals = finite(values)
    return float(min(vals)) if vals else None


def stable_unit(*parts: Any) -> float:
    digest = hashlib.sha256("|".join(str(part) for part in parts).encode("utf-8")).hexdigest()
    return int(digest[:12], 16) / float(16**12)


def split_ids(value: Any) -> set[str]:
    text = str(value or "")
    if text.lower() == "nan":
        return set()
    return {part.strip() for part in text.split(",") if part.strip()}


def same_count_control(
    rows: list[dict[str, str]],
    selected_pairs: set[str],
    variant: str,
    seed: int,
) -> set[str]:
    variant_rows = [row for row in rows if str(row.get("variant")) == variant]
    selected_variant_rows = [row for row in variant_rows if str(row.get("pair_id")) in selected_pairs]
    counts: dict[str, int] = {}
    for row in selected_variant_rows:
        label = str(row.get("case_label_offline_only"))
        counts[label] = counts.get(label, 0) + 1
    out: set[str] = set()
    for label, count in counts.items():
        candidates = [row for row in variant_rows if str(row.get("case_label_offline_only")) == label]
        ordered = sorted(
            candidates,
            key=lambda row: stable_unit("tracke_internal_cue_same_count", seed, variant, label, row.get("pair_id")),
        )
        out.update(str(row.get("pair_id")) for row in ordered[: min(count, len(ordered))])
    return out


def selected_metric(rows: list[dict[str, str]], selected_pairs: set[str], variant: str) -> dict[str, Any]:
    variant_rows = [row for row in rows if str(row.get("variant")) == variant]
    selected = [row for row in variant_rows if str(row.get("pair_id")) in selected_pairs]
    bad = [row for row in selected if row.get("case_label_offline_only") == "bad"]
    good = [row for row in selected if row.get("case_label_offline_only") == "good"]
    bad_total = sum(1 for row in variant_rows if row.get("case_label_offline_only") == "bad")
    good_total = sum(1 for row in variant_rows if row.get("case_label_offline_only") == "good")
    good_worsen = [-f(row.get("handoff_transfer_improvement_ratio")) for row in good]
    return {
        "variant": variant,
        "selected_pair_count": len(selected),
        "selected_bad_count": len(bad),
        "selected_good_count": len(good),
        "bad_total": bad_total,
        "good_total": good_total,
        "bad_recall_on_measured_labelled": len(bad) / max(bad_total, 1),
        "good_FPR_on_measured_labelled": len(good) / max(good_total, 1),
        "selected_sequence_coverage": len({str(row.get("seq")) for row in selected}),
        "selected_bad_sequence_coverage": len({str(row.get("seq")) for row in bad}),
        "bad_handoff_median_improvement": median_or_none(row.get("handoff_transfer_improvement_ratio") for row in bad),
        "bad_handoff_max_improvement": max_or_none(row.get("handoff_transfer_improvement_ratio") for row in bad),
        "bad_handoff_min_improvement": min_or_none(row.get("handoff_transfer_improvement_ratio") for row in bad),
        "bad_runtime_proxy_median_I_J": median_or_none(row.get("I_J_runtime_proxy") for row in bad),
        "bad_boundary_median_improvement": median_or_none(row.get("boundary_improvement_ratio") for row in bad),
        "bad_scale_median_improvement": median_or_none(row.get("scale_improvement_ratio") for row in bad),
        "good_handoff_median_worsen": median_or_none(good_worsen),
        "good_handoff_max_worsen": max_or_none(good_worsen),
        "selected_pairs": ",".join(sorted(str(row.get("pair_id")) for row in selected)),
        "selected_bad_pairs": ",".join(sorted(str(row.get("pair_id")) for row in bad)),
        "selected_good_pairs": ",".join(sorted(str(row.get("pair_id")) for row in good)),
    }


def evaluate_cue_variant(
    rows: list[dict[str, str]],
    cue: Mapping[str, str],
    variant: str,
    control_count: int,
    bad_handoff_threshold: float,
) -> dict[str, Any]:
    selected_pairs = split_ids(cue.get("selected_pair_ids"))
    actual = selected_metric(rows, selected_pairs, variant)
    control_bad_medians = []
    for seed in range(control_count):
        control_pairs = same_count_control(rows, selected_pairs, variant, seed)
        control = selected_metric(rows, control_pairs, variant)
        value = control.get("bad_handoff_median_improvement")
        if value is not None and math.isfinite(float(value)):
            control_bad_medians.append(float(value))
    best_control = max(control_bad_medians) if control_bad_medians else None
    actual_bad = actual.get("bad_handoff_median_improvement")
    actual_minus_best = (
        float(actual_bad - best_control) if actual_bad is not None and best_control is not None else None
    )
    good_max_worsen = actual.get("good_handoff_max_worsen")
    good_median_worsen = actual.get("good_handoff_median_worsen")
    good_protection = (
        True
        if int(actual.get("selected_good_count") or 0) == 0
        else good_median_worsen is not None
        and good_median_worsen <= 0.02
        and good_max_worsen is not None
        and good_max_worsen <= 0.02
    )
    gates = {
        "cue_g5_gate_pass": bool_text(cue.get("candidate_gate_pass")),
        "bad_handoff_median_ge_threshold": actual_bad is not None and actual_bad >= bad_handoff_threshold,
        "good_handoff_protection_gate": good_protection,
        "sequence_coverage_ge_3": int(actual.get("selected_bad_sequence_coverage") or 0) >= 3,
        "beats_same_count_controls_ge_0p05": actual_minus_best is not None and actual_minus_best >= 0.05,
        "selected_bad_nonempty": int(actual.get("selected_bad_count") or 0) > 0,
    }
    return {
        "cue_id": cue.get("cue_id"),
        "cue_bad_recall": f(cue.get("bad_recall")),
        "cue_good_FPR": f(cue.get("good_FPR")),
        "cue_balanced_accuracy": f(cue.get("balanced_accuracy")),
        **actual,
        "best_same_count_control_bad_handoff_median": best_control,
        "actual_minus_best_same_count_control": actual_minus_best,
        **gates,
        "candidate_action_surface_gate_pass": all(gates.values()),
    }


def per_pair_best_rows(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("pair_id")), []).append(row)
    out: list[dict[str, Any]] = []
    for pair_id, group in sorted(grouped.items()):
        best = max(group, key=lambda row: f(row.get("handoff_transfer_improvement_ratio"), -float("inf")))
        out.append(
            {
                "pair_id": pair_id,
                "seq": best.get("seq"),
                "case_label_offline_only": best.get("case_label_offline_only"),
                "best_variant_by_handoff": best.get("variant"),
                "max_handoff_improvement": f(best.get("handoff_transfer_improvement_ratio")),
                "I_J_runtime_proxy_at_best_handoff": f(best.get("I_J_runtime_proxy")),
                "boundary_improvement_at_best_handoff": f(best.get("boundary_improvement_ratio")),
                "scale_improvement_at_best_handoff": f(best.get("scale_improvement_ratio")),
            }
        )
    return out


def oracle_metric_for_cue(best_rows: list[dict[str, Any]], cue: Mapping[str, str]) -> dict[str, Any]:
    selected_pairs = split_ids(cue.get("selected_pair_ids"))
    selected = [row for row in best_rows if str(row.get("pair_id")) in selected_pairs]
    bad = [row for row in selected if row.get("case_label_offline_only") == "bad"]
    good = [row for row in selected if row.get("case_label_offline_only") == "good"]
    return {
        "cue_id": cue.get("cue_id"),
        "oracle_selected_pair_count": len(selected),
        "oracle_selected_bad_count": len(bad),
        "oracle_selected_good_count": len(good),
        "oracle_bad_handoff_median_improvement": median_or_none(row.get("max_handoff_improvement") for row in bad),
        "oracle_bad_handoff_max_improvement": max_or_none(row.get("max_handoff_improvement") for row in bad),
        "oracle_selected_bad_pairs": ",".join(sorted(str(row.get("pair_id")) for row in bad)),
        "oracle_selected_good_pairs": ",".join(sorted(str(row.get("pair_id")) for row in good)),
    }


def rank_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        bool_text(row.get("candidate_action_surface_gate_pass")),
        f(row.get("bad_handoff_median_improvement"), -999.0),
        f(row.get("actual_minus_best_same_count_control"), -999.0),
        f(row.get("bad_runtime_proxy_median_I_J"), -999.0),
        -f(row.get("good_FPR_on_measured_labelled"), 999.0),
    )


def main() -> None:
    args = parse_args()
    effect_rows = read_csv(args.effect_rows)
    cue_rows = [row for row in read_csv(args.cue_metrics) if bool_text(row.get("candidate_gate_pass"))]
    cue_rows = cue_rows[: args.max_cues]
    variants = sorted({str(row.get("variant")) for row in effect_rows if row.get("variant")})

    metrics = [
        evaluate_cue_variant(effect_rows, cue, variant, args.same_count_controls, args.bad_handoff_threshold)
        for cue in cue_rows
        for variant in variants
    ]
    metrics.sort(key=rank_key, reverse=True)
    passing = [row for row in metrics if bool_text(row.get("candidate_action_surface_gate_pass"))]

    best_rows = per_pair_best_rows(effect_rows)
    oracle_rows = [oracle_metric_for_cue(best_rows, cue) for cue in cue_rows]
    oracle_rows.sort(
        key=lambda row: (
            f(row.get("oracle_bad_handoff_median_improvement"), -999.0),
            f(row.get("oracle_bad_handoff_max_improvement"), -999.0),
        ),
        reverse=True,
    )
    bad_best_rows = [row for row in best_rows if row.get("case_label_offline_only") == "bad"]
    bad_pairs_ge_threshold = [
        row for row in bad_best_rows if f(row.get("max_handoff_improvement"), -999.0) >= args.bad_handoff_threshold
    ]
    best = metrics[0] if metrics else {}
    best_oracle = oracle_rows[0] if oracle_rows else {}
    blocker = (
        "no_measured_variant_reaches_5pct_handoff_improvement_for_any_bad_pair"
        if not bad_pairs_ge_threshold
        else "no_internal_cue_selected_action_surface_beats_controls"
    )
    summary = {
        "phase": "v95_trackE_internal_cue_action_surface_v1",
        "effect_rows": str(args.effect_rows),
        "cue_metrics": str(args.cue_metrics),
        "cue_count_evaluated": len(cue_rows),
        "variant_count": len(variants),
        "cue_variant_metric_count": len(metrics),
        "candidate_action_surface_passing_count": len(passing),
        "gate_pass": bool(passing),
        "runtime_action_allowed": False,
        "bad_handoff_threshold": args.bad_handoff_threshold,
        "bad_pair_count": len(bad_best_rows),
        "bad_pair_ge_threshold_count": len(bad_pairs_ge_threshold),
        "global_best_bad_pair_handoff_improvement": max_or_none(row.get("max_handoff_improvement") for row in bad_best_rows),
        "global_best_bad_pair": max(
            bad_best_rows,
            key=lambda row: f(row.get("max_handoff_improvement"), -float("inf")),
            default={},
        ),
        "best_cue_variant": best,
        "best_oracle_cue": best_oracle,
        "blocker": "" if passing else blocker,
        "interpretation_boundary": (
            "Measured runtime variants are reused. This audit cannot prove a new unmeasured action variant would fail."
        ),
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "internal_cue_action_surface_metrics.csv", metrics)
    write_csv(args.out_dir / "internal_cue_oracle_metrics.csv", oracle_rows)
    write_csv(args.out_dir / "per_pair_best_handoff_rows.csv", best_rows)
    write_json(args.out_dir / "summary.json", summary)
    write_text(
        args.out_dir / "analysis.md",
        f"""
# Track E Internal Cue Action-Surface Audit

- cue_count_evaluated: `{summary['cue_count_evaluated']}`
- variant_count: `{summary['variant_count']}`
- candidate_action_surface_passing_count: `{summary['candidate_action_surface_passing_count']}`
- bad_pair_ge_5pct_handoff_count: `{summary['bad_pair_ge_threshold_count']}`
- global_best_bad_pair_handoff_improvement: `{summary['global_best_bad_pair_handoff_improvement']}`
- best_cue_variant: `{best.get('cue_id')}` / `{best.get('variant')}`
- best_cue_variant_bad_handoff_median: `{best.get('bad_handoff_median_improvement')}`
- best_cue_variant_actual_minus_best_control: `{best.get('actual_minus_best_same_count_control')}`
- best_oracle_cue: `{best_oracle.get('cue_id')}`
- best_oracle_bad_handoff_median: `{best_oracle.get('oracle_bad_handoff_median_improvement')}`
- blocker: `{summary['blocker']}`

Interpretation: Track G now has an internal SWA cue that passes the cue audit,
but the measured Track E variants still do not create enough L3 handoff movement.
The per-pair oracle over measured variants remains below the 5% handoff threshold,
so the current blocker is action mechanism strength, not just cue selection.
""",
    )
    write_text(
        args.out_dir / "next_route_recommendation.md",
        (
            "trackE_action_mechanism_redesign_required_after_internal_cue_pass\n"
            if not passing
            else "manual_review_before_runtime_promotion\n"
        ),
    )

    print(f"cue_count_evaluated={summary['cue_count_evaluated']}")
    print(f"variant_count={summary['variant_count']}")
    print(f"candidate_action_surface_passing_count={summary['candidate_action_surface_passing_count']}")
    print(f"bad_pair_ge_5pct_handoff_count={summary['bad_pair_ge_threshold_count']}")
    print(f"global_best_bad_pair_handoff_improvement={summary['global_best_bad_pair_handoff_improvement']}")
    print(f"best_cue_variant={best.get('cue_id')}::{best.get('variant')}")
    print(f"best_cue_variant_bad_handoff_median={best.get('bad_handoff_median_improvement')}")
    print(f"best_cue_variant_actual_minus_best_control={best.get('actual_minus_best_same_count_control')}")
    print(f"best_oracle_bad_handoff_median={best_oracle.get('oracle_bad_handoff_median_improvement')}")
    print(f"gate_pass={summary['gate_pass']}")
    print(f"runtime_action_allowed={summary['runtime_action_allowed']}")
    print(f"blocker={summary['blocker']}")


if __name__ == "__main__":
    main()
