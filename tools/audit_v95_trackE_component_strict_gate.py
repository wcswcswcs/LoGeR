#!/usr/bin/env python3
"""Audit Track E runtime probe variants with component-level gates.

The mixed I_J proxy in ``build_v94_runtime_probe_sensitivity.py`` includes
handoff transfer, scale jump, boundary update norm, and merge residual terms.
For v95 Track E, the plan requires that SWA transport actually improve L3 /
handoff behavior while protecting good controls. This audit makes that stricter
requirement explicit instead of letting a boundary-only no-op dominate.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean, median
from typing import Any


DEFAULT_EFFECT_ROWS = Path(
    "results/acl2_v95tf_multiroute_semantic_memory_evidence_control/"
    "trackE_alpha04_runtime_probe/runtime_probe_effect_rows.csv"
)
DEFAULT_OUT_DIR = Path(
    "results/acl2_v95tf_multiroute_semantic_memory_evidence_control/"
    "trackE_component_strict_gate"
)


def f(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def finite(values: list[Any]) -> list[float]:
    return [v for v in (f(item) for item in values) if math.isfinite(v)]


def median_or_none(values: list[float]) -> float | None:
    return float(median(values)) if values else None


def mean_or_none(values: list[float]) -> float | None:
    return float(mean(values)) if values else None


def max_or_none(values: list[float]) -> float | None:
    return float(max(values)) if values else None


def min_or_none(values: list[float]) -> float | None:
    return float(min(values)) if values else None


def load_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def variant_metrics(rows: list[dict[str, Any]], variant: str) -> dict[str, Any]:
    group = [row for row in rows if str(row.get("variant")) == variant]
    bad = [row for row in group if row.get("case_label_offline_only") == "bad"]
    good = [row for row in group if row.get("case_label_offline_only") == "good"]

    bad_handoff = finite([row.get("handoff_transfer_improvement_ratio") for row in bad])
    good_handoff = finite([row.get("handoff_transfer_improvement_ratio") for row in good])
    bad_scale = finite([row.get("scale_improvement_ratio") for row in bad])
    good_scale = finite([row.get("scale_improvement_ratio") for row in good])
    bad_boundary = finite([row.get("boundary_improvement_ratio") for row in bad])
    good_boundary = finite([row.get("boundary_improvement_ratio") for row in good])
    bad_proxy = finite([row.get("I_J_runtime_proxy") for row in bad])
    good_proxy_w = finite([row.get("W_good_runtime_proxy") for row in good])
    good_handoff_worsen = [-value for value in good_handoff]

    bad_handoff_median = median_or_none(bad_handoff)
    good_handoff_worsen_median = median_or_none(good_handoff_worsen)
    good_handoff_worsen_max = max_or_none(good_handoff_worsen)

    strict_gates = {
        "bad_handoff_median_improvement_ge_0p05": (
            bad_handoff_median is not None and bad_handoff_median >= 0.05
        ),
        "good_handoff_median_worsen_le_0p02": (
            good_handoff_worsen_median is not None and good_handoff_worsen_median <= 0.02
        ),
        "good_handoff_max_worsen_le_0p02": (
            good_handoff_worsen_max is not None and good_handoff_worsen_max <= 0.02
        ),
        "bad_and_good_rows_present": bool(bad) and bool(good),
        "sequence_coverage_ge_3": len({row.get("seq") for row in group}) >= 3,
    }
    strict_gate_pass = all(strict_gates.values())

    boundary_only_warning = False
    if bad_handoff_median is not None and bad_boundary:
        bad_boundary_median = median_or_none(bad_boundary)
        bad_proxy_median = median_or_none(bad_proxy)
        boundary_only_warning = bool(
            bad_proxy_median is not None
            and bad_boundary_median is not None
            and bad_proxy_median >= 0.05
            and bad_boundary_median >= 0.25
            and bad_handoff_median < 0.05
        )

    return {
        "variant": variant,
        "row_count": len(group),
        "sequence_coverage": len({row.get("seq") for row in group}),
        "bad_rows": len(bad),
        "good_rows": len(good),
        "bad_handoff_median_improvement": bad_handoff_median,
        "bad_handoff_mean_improvement": mean_or_none(bad_handoff),
        "bad_handoff_min_improvement": min_or_none(bad_handoff),
        "bad_handoff_max_improvement": max_or_none(bad_handoff),
        "good_handoff_median_worsen": good_handoff_worsen_median,
        "good_handoff_mean_worsen": mean_or_none(good_handoff_worsen),
        "good_handoff_max_worsen": good_handoff_worsen_max,
        "bad_scale_median_improvement": median_or_none(bad_scale),
        "good_scale_median_improvement": median_or_none(good_scale),
        "bad_boundary_median_improvement": median_or_none(bad_boundary),
        "good_boundary_median_improvement": median_or_none(good_boundary),
        "bad_proxy_median_I_J": median_or_none(bad_proxy),
        "good_proxy_median_worsen": median_or_none(good_proxy_w),
        "strict_gate_pass": strict_gate_pass,
        "boundary_only_warning": boundary_only_warning,
        **strict_gates,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--effect-rows", type=Path, default=DEFAULT_EFFECT_ROWS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    rows = load_rows(args.effect_rows)
    variants = sorted({str(row.get("variant")) for row in rows if row.get("variant")})
    metrics = [variant_metrics(rows, variant) for variant in variants]
    metrics.sort(
        key=lambda row: (
            bool(row.get("strict_gate_pass")),
            f(row.get("bad_handoff_median_improvement")),
        ),
        reverse=True,
    )

    passing = [row for row in metrics if row.get("strict_gate_pass")]
    boundary_warnings = [row for row in metrics if row.get("boundary_only_warning")]
    best = metrics[0] if metrics else {}
    summary = {
        "phase": "v95_trackE_component_strict_gate",
        "effect_rows": str(args.effect_rows),
        "variant_count": len(metrics),
        "row_count": len(rows),
        "strict_gate_pass_count": len(passing),
        "strict_gate_pass": bool(passing),
        "best_variant": best.get("variant"),
        "best_bad_handoff_median_improvement": best.get("bad_handoff_median_improvement"),
        "best_good_handoff_max_worsen": best.get("good_handoff_max_worsen"),
        "boundary_only_warning_variants": [row.get("variant") for row in boundary_warnings],
        "blocker": "" if passing else "no_variant_passes_component_strict_handoff_gate",
        "runtime_action_allowed": False,
        "thresholds": {
            "bad_handoff_median_improvement_min": 0.05,
            "good_handoff_median_worsen_max": 0.02,
            "good_handoff_max_worsen_max": 0.02,
            "sequence_coverage_min": 3,
        },
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "component_strict_variant_metrics.csv", metrics)
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.out_dir / "analysis.md").write_text(
        "\n".join(
            [
                "# Track E Component Strict Gate Audit",
                "",
                f"- effect_rows: `{args.effect_rows}`",
                f"- variants: `{len(metrics)}`",
                f"- strict_gate_pass_count: `{len(passing)}`",
                f"- best_variant: `{summary['best_variant']}`",
                f"- best_bad_handoff_median_improvement: `{summary['best_bad_handoff_median_improvement']}`",
                f"- best_good_handoff_max_worsen: `{summary['best_good_handoff_max_worsen']}`",
                f"- boundary_only_warning_variants: `{summary['boundary_only_warning_variants']}`",
                f"- blocker: `{summary['blocker']}`",
                "",
                "Interpretation: Track E action remains blocked unless a variant improves",
                "bad handoff transfer by at least 5% while keeping good-control handoff",
                "worsen at or below 2%. Mixed proxy gains from boundary update suppression",
                "are diagnostic only.",
                "",
            ]
        ),
        encoding="utf-8",
    )

    print(f"variant_count={summary['variant_count']}")
    print(f"strict_gate_pass_count={summary['strict_gate_pass_count']}")
    print(f"best_variant={summary['best_variant']}")
    print(f"best_bad_handoff_median_improvement={summary['best_bad_handoff_median_improvement']}")
    print(f"blocker={summary['blocker']}")


if __name__ == "__main__":
    main()
