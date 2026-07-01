#!/usr/bin/env python3
"""Audit measured Track E eligibility upper bounds.

This is a diagnostic-only audit for the v95 Track E fail-forward path. It
answers a narrow question: if delay/reject/transmit eligibility were allowed to
choose from every already measured runtime-probe action row, is there enough
measured L3 handoff movement to satisfy the 5% action threshold?

The oracle policies here are upper bounds over measured rows. They are not
runtime policies and cannot prove that an unmeasured action mechanism would
fail.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import median
from typing import Any, Iterable, Mapping


ROOT = Path("results/acl2_v95tf_multiroute_semantic_memory_evidence_control")
DEFAULT_CUE_METRICS = ROOT / "trackG_swa_internal_cue_eval_v1/method_safe_internal_candidate_metrics.csv"
DEFAULT_OUT_DIR = ROOT / "trackE_measured_eligibility_upper_bound_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--effect-rows", type=Path, nargs="+", required=True)
    parser.add_argument("--cue-metrics", type=Path, default=DEFAULT_CUE_METRICS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--bad-handoff-threshold", type=float, default=0.05)
    parser.add_argument("--max-cues", type=int, default=120)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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


def split_ids(value: Any) -> set[str]:
    text = str(value or "")
    if text.lower() == "nan":
        return set()
    return {part.strip() for part in text.split(",") if part.strip()}


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


def load_effect_rows(paths: list[Path]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for path in paths:
        for row in read_csv(path):
            enriched = dict(row)
            enriched["source_path"] = str(path)
            enriched["action_id"] = f"{path.parent.name}::{row.get('variant')}"
            out.append(enriched)
    return out


def best_by_pair(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        pair_id = str(row.get("pair_id") or "")
        if pair_id:
            grouped.setdefault(pair_id, []).append(row)

    best_rows: list[dict[str, Any]] = []
    for pair_id, group in sorted(grouped.items()):
        best = max(group, key=lambda row: f(row.get("handoff_transfer_improvement_ratio"), -float("inf")))
        best_value = f(best.get("handoff_transfer_improvement_ratio"))
        best_rows.append(
            {
                "pair_id": pair_id,
                "seq": best.get("seq"),
                "prev_chunk": best.get("prev_chunk"),
                "curr_chunk": best.get("curr_chunk"),
                "case_label_offline_only": best.get("case_label_offline_only"),
                "best_action_id": best.get("action_id"),
                "best_variant": best.get("variant"),
                "best_source_path": best.get("source_path"),
                "best_handoff_improvement": best_value,
                "reject_negative_handoff_improvement": max(0.0, best_value) if math.isfinite(best_value) else None,
                "I_J_runtime_proxy_at_best": f(best.get("I_J_runtime_proxy")),
                "scale_improvement_at_best": f(best.get("scale_improvement_ratio")),
                "boundary_improvement_at_best": f(best.get("boundary_improvement_ratio")),
            }
        )
    return best_rows


def policy_metrics(rows: list[dict[str, Any]], selected_pairs: set[str], policy: str) -> dict[str, Any]:
    selected = [row for row in rows if str(row.get("pair_id")) in selected_pairs]
    bad = [row for row in selected if row.get("case_label_offline_only") == "bad"]
    good = [row for row in selected if row.get("case_label_offline_only") == "good"]
    field = "best_handoff_improvement" if policy == "transmit_best_measured" else "reject_negative_handoff_improvement"
    good_worsen = [-f(row.get(field)) for row in good]
    return {
        "policy": policy,
        "selected_pair_count": len(selected),
        "selected_bad_count": len(bad),
        "selected_good_count": len(good),
        "selected_sequence_coverage": len({str(row.get("seq")) for row in selected}),
        "selected_bad_sequence_coverage": len({str(row.get("seq")) for row in bad}),
        "selected_pairs": ",".join(sorted(str(row.get("pair_id")) for row in selected)),
        "selected_bad_pairs": ",".join(sorted(str(row.get("pair_id")) for row in bad)),
        "selected_good_pairs": ",".join(sorted(str(row.get("pair_id")) for row in good)),
        "bad_handoff_median_improvement": median_or_none(row.get(field) for row in bad),
        "bad_handoff_max_improvement": max_or_none(row.get(field) for row in bad),
        "bad_handoff_min_improvement": min_or_none(row.get(field) for row in bad),
        "good_handoff_median_worsen": median_or_none(good_worsen),
        "good_handoff_max_worsen": max_or_none(good_worsen),
    }


def cue_policy_rows(
    best_rows: list[dict[str, Any]],
    cue_rows: list[dict[str, str]],
    threshold: float,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for cue in cue_rows:
        selected_pairs = split_ids(cue.get("selected_pair_ids"))
        for policy in ("transmit_best_measured", "reject_negative_best_measured"):
            metrics = policy_metrics(best_rows, selected_pairs, policy)
            bad_median = metrics.get("bad_handoff_median_improvement")
            bad_max = metrics.get("bad_handoff_max_improvement")
            good_max_worsen = metrics.get("good_handoff_max_worsen")
            good_protection = (
                int(metrics.get("selected_good_count") or 0) == 0
                or (good_max_worsen is not None and good_max_worsen <= 0.02)
            )
            out.append(
                {
                    "cue_id": cue.get("cue_id"),
                    "cue_bad_recall": f(cue.get("bad_recall")),
                    "cue_good_FPR": f(cue.get("good_FPR")),
                    "cue_balanced_accuracy": f(cue.get("balanced_accuracy")),
                    "cue_candidate_gate_pass": bool_text(cue.get("candidate_gate_pass")),
                    **metrics,
                    "bad_handoff_median_ge_threshold": bad_median is not None and bad_median >= threshold,
                    "bad_handoff_max_ge_threshold": bad_max is not None and bad_max >= threshold,
                    "good_handoff_protection_gate": good_protection,
                    "sequence_coverage_ge_3": int(metrics.get("selected_bad_sequence_coverage") or 0) >= 3,
                    "diagnostic_upper_bound_gate_pass": (
                        bad_median is not None
                        and bad_median >= threshold
                        and good_protection
                        and int(metrics.get("selected_bad_sequence_coverage") or 0) >= 3
                    ),
                }
            )
    return out


def main() -> None:
    args = parse_args()
    rows = load_effect_rows(args.effect_rows)
    cue_rows = [row for row in read_csv(args.cue_metrics) if bool_text(row.get("candidate_gate_pass"))]
    cue_rows = cue_rows[: args.max_cues]
    best_rows = best_by_pair(rows)
    cue_metrics = cue_policy_rows(best_rows, cue_rows, args.bad_handoff_threshold)
    cue_metrics.sort(
        key=lambda row: (
            bool(row.get("diagnostic_upper_bound_gate_pass")),
            f(row.get("bad_handoff_median_improvement"), -999.0),
            f(row.get("bad_handoff_max_improvement"), -999.0),
        ),
        reverse=True,
    )

    bad_best = [row for row in best_rows if row.get("case_label_offline_only") == "bad"]
    good_best = [row for row in best_rows if row.get("case_label_offline_only") == "good"]
    measured_ge_threshold = [
        row for row in best_rows if f(row.get("best_handoff_improvement"), -999.0) >= args.bad_handoff_threshold
    ]
    bad_ge_threshold = [
        row for row in bad_best if f(row.get("best_handoff_improvement"), -999.0) >= args.bad_handoff_threshold
    ]
    passing = [row for row in cue_metrics if row.get("diagnostic_upper_bound_gate_pass")]
    best_cue = cue_metrics[0] if cue_metrics else {}
    global_best = max(
        best_rows,
        key=lambda row: f(row.get("best_handoff_improvement"), -float("inf")),
        default={},
    )
    global_best_bad = max(
        bad_best,
        key=lambda row: f(row.get("best_handoff_improvement"), -float("inf")),
        default={},
    )
    summary = {
        "phase": "v95_trackE_measured_eligibility_upper_bound_v1",
        "effect_rows": [str(path) for path in args.effect_rows],
        "cue_metrics": str(args.cue_metrics),
        "measured_row_count": len(rows),
        "unique_pair_count": len(best_rows),
        "unique_bad_pair_count": len(bad_best),
        "unique_good_pair_count": len(good_best),
        "unique_action_count": len({row.get("action_id") for row in rows}),
        "cue_count_evaluated": len(cue_rows),
        "bad_handoff_threshold": args.bad_handoff_threshold,
        "global_best_handoff_improvement": global_best.get("best_handoff_improvement"),
        "global_best_pair": global_best,
        "global_best_bad_handoff_improvement": global_best_bad.get("best_handoff_improvement"),
        "global_best_bad_pair": global_best_bad,
        "measured_pair_ge_threshold_count": len(measured_ge_threshold),
        "bad_pair_ge_threshold_count": len(bad_ge_threshold),
        "diagnostic_upper_bound_passing_count": len(passing),
        "best_cue_policy": best_cue,
        "gate_pass": bool(passing),
        "runtime_action_allowed": False,
        "blocker": (
            ""
            if passing
            else "measured_action_pool_has_no_5pct_handoff_headroom_for_bad_pairs"
            if not bad_ge_threshold
            else "cue_eligibility_policy_does_not_convert_measured_headroom_into_passing_surface"
        ),
        "interpretation_boundary": (
            "Oracle reject/best policies are measured-pool upper bounds only; "
            "they are not method-safe runtime policies and cannot reject unmeasured mechanisms."
        ),
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "per_pair_best_measured_handoff.csv", best_rows)
    write_csv(args.out_dir / "cue_policy_upper_bound_metrics.csv", cue_metrics)
    write_json(args.out_dir / "summary.json", summary)
    write_text(
        args.out_dir / "analysis.md",
        f"""
# Track E Measured Eligibility Upper-Bound Audit

- measured_row_count: `{summary['measured_row_count']}`
- unique_pair_count: `{summary['unique_pair_count']}`
- unique_bad_pair_count: `{summary['unique_bad_pair_count']}`
- unique_action_count: `{summary['unique_action_count']}`
- global_best_handoff_improvement: `{summary['global_best_handoff_improvement']}`
- global_best_bad_handoff_improvement: `{summary['global_best_bad_handoff_improvement']}`
- measured_pair_ge_5pct_count: `{summary['measured_pair_ge_threshold_count']}`
- bad_pair_ge_5pct_count: `{summary['bad_pair_ge_threshold_count']}`
- diagnostic_upper_bound_passing_count: `{summary['diagnostic_upper_bound_passing_count']}`
- best_cue_policy: `{best_cue.get('cue_id')}` / `{best_cue.get('policy')}`
- best_cue_bad_handoff_median: `{best_cue.get('bad_handoff_median_improvement')}`
- best_cue_bad_handoff_max: `{best_cue.get('bad_handoff_max_improvement')}`
- blocker: `{summary['blocker']}`

Interpretation: even an oracle over already measured actions, with optional
reject-negative behavior, does not create a 5% L3 handoff improvement surface
for the bad pairs. This supports returning from eligibility tuning to action
mechanism redesign or source diagnosis. It does not rule out unmeasured actions.
""",
    )

    print(f"measured_row_count={summary['measured_row_count']}")
    print(f"unique_pair_count={summary['unique_pair_count']}")
    print(f"unique_action_count={summary['unique_action_count']}")
    print(f"global_best_handoff_improvement={summary['global_best_handoff_improvement']}")
    print(f"global_best_bad_handoff_improvement={summary['global_best_bad_handoff_improvement']}")
    print(f"measured_pair_ge_5pct_count={summary['measured_pair_ge_threshold_count']}")
    print(f"bad_pair_ge_5pct_count={summary['bad_pair_ge_threshold_count']}")
    print(f"diagnostic_upper_bound_passing_count={summary['diagnostic_upper_bound_passing_count']}")
    print(f"best_cue_policy={best_cue.get('cue_id')}::{best_cue.get('policy')}")
    print(f"best_cue_bad_handoff_median={best_cue.get('bad_handoff_median_improvement')}")
    print(f"best_cue_bad_handoff_max={best_cue.get('bad_handoff_max_improvement')}")
    print(f"gate_pass={summary['gate_pass']}")
    print(f"runtime_action_allowed={summary['runtime_action_allowed']}")
    print(f"blocker={summary['blocker']}")


if __name__ == "__main__":
    main()
