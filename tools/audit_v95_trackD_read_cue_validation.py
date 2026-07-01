#!/usr/bin/env python3
"""Validate v95 Track D READ cue rows against READ_LOCAL controls.

This audit is diagnostic-only. It uses existing read-cue mass rows and offline
case-bank labels to test whether any simple READ cue can separate
READ_LOCAL_BAD cases from good controls with same-count random controls. It
does not perform a READ action and does not claim QK compatibility.
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
DEFAULT_READ_ROWS = ROOT / "trackD_read_eligibility/rows.csv"
DEFAULT_CASE_ROWS = ROOT / "trackA_base_case_bank/rows.csv"
DEFAULT_OUT_DIR = ROOT / "trackD_read_cue_validation_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--read-rows", type=Path, default=DEFAULT_READ_ROWS)
    parser.add_argument("--case-rows", type=Path, default=DEFAULT_CASE_ROWS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--random-seeds", type=int, default=256)
    parser.add_argument("--max-candidates", type=int, default=200)
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


def finite(values: Iterable[Any]) -> list[float]:
    return [value for value in (f(item) for item in values) if math.isfinite(value)]


def quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    vals = sorted(values)
    pos = (len(vals) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return vals[lo]
    return vals[lo] * (hi - pos) + vals[hi] * (pos - lo)


def stable_unit(*parts: Any) -> float:
    digest = hashlib.sha256("|".join(str(part) for part in parts).encode("utf-8")).hexdigest()
    return int(digest[:12], 16) / float(16**12)


def stable_random_mask(rows: list[Mapping[str, Any]], count: int, seed: int) -> list[bool]:
    order = sorted(range(len(rows)), key=lambda idx: stable_unit("v95_trackD_read_random", seed, rows[idx].get("pair_id"), idx))
    selected = set(order[: min(count, len(order))])
    return [idx in selected for idx in range(len(rows))]


def seq_count_random_mask(rows: list[Mapping[str, Any]], mask: list[bool], seed: int) -> list[bool]:
    by_seq: dict[str, list[int]] = {}
    for idx, row in enumerate(rows):
        by_seq.setdefault(str(row.get("seq")), []).append(idx)
    selected: set[int] = set()
    for seq, indices in by_seq.items():
        count = sum(1 for idx in indices if mask[idx])
        order = sorted(indices, key=lambda idx: stable_unit("v95_trackD_read_seq_random", seed, seq, rows[idx].get("pair_id"), idx))
        selected.update(order[: min(count, len(order))])
    return [idx in selected for idx in range(len(rows))]


def join_rows(read_rows: list[dict[str, str]], case_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    by_id = {str(row.get("case_id")): row for row in case_rows}
    out: list[dict[str, Any]] = []
    for row in read_rows:
        pair_id = str(row.get("pair_id"))
        case = by_id.get(pair_id, {})
        merged = dict(row)
        merged.update(
            {
                "case_id": pair_id,
                "v95_case_bucket": case.get("v95_case_bucket", ""),
                "case_label_offline_only": case.get("case_label_offline_only", ""),
                "failure_type_primary": case.get("failure_type_primary", ""),
                "L2_intra_scale_cv": case.get("L2_intra_scale_cv", ""),
                "L2_head_tail_proxy_error": case.get("L2_head_tail_proxy_error", ""),
                "L3_J_handoff": case.get("L3_J_handoff", ""),
            }
        )
        out.append(merged)
    return out


def add_derived_scores(rows: list[dict[str, Any]]) -> None:
    for row in rows:
        active = f(row.get("read_active_mass"), 0.0)
        stable = f(row.get("read_stable_mass"), 0.0)
        invalid = f(row.get("read_invalid_mass"), 0.0)
        context = f(row.get("read_context_mass"), 0.0)
        shuffle = f(row.get("read_semantic_shuffle_mass"), 0.0)
        random_same = f(row.get("read_same_mass_random_mass"), 0.0)
        entropy = f(row.get("read_query_entropy"), 0.0)
        row["read_invalid_plus_context_mass"] = invalid + context
        row["read_invalid_over_active"] = invalid / max(active, 1e-12)
        row["read_context_over_active"] = context / max(active, 1e-12)
        row["read_unstable_over_stable"] = (invalid + context) / max(stable, 1e-12)
        row["read_active_minus_shuffle"] = active - shuffle
        row["read_active_minus_same_mass_random"] = active - random_same
        row["read_invalid_minus_stable"] = invalid - stable
        row["read_context_minus_stable"] = context - stable
        row["read_entropy"] = entropy


def build_candidates(rows: list[Mapping[str, Any]], max_candidates: int) -> list[dict[str, Any]]:
    high_features = [
        "read_active_mass",
        "read_invalid_mass",
        "read_context_mass",
        "read_invalid_plus_context_mass",
        "read_invalid_over_active",
        "read_context_over_active",
        "read_unstable_over_stable",
        "read_active_minus_shuffle",
        "read_active_minus_same_mass_random",
        "read_invalid_minus_stable",
        "read_context_minus_stable",
        "read_entropy",
    ]
    low_features = ["read_stable_mass"]
    candidates: list[dict[str, Any]] = []
    for feature in high_features:
        vals = finite(row.get(feature) for row in rows)
        for q in (0.50, 0.60, 0.70, 0.75, 0.80, 0.90):
            threshold = quantile(vals, q)
            if threshold is not None:
                candidates.append({"cue_id": f"{feature.upper()}_GE_Q{int(q*100)}", "feature": feature, "direction": "ge", "threshold": threshold})
    for feature in low_features:
        vals = finite(row.get(feature) for row in rows)
        for q in (0.10, 0.20, 0.25, 0.30, 0.40, 0.50):
            threshold = quantile(vals, q)
            if threshold is not None:
                candidates.append({"cue_id": f"{feature.upper()}_LE_Q{int(q*100)}", "feature": feature, "direction": "le", "threshold": threshold})

    # A few compact two-term masks reflecting "unreliable read, weak stable support".
    vals_invalid = finite(row.get("read_invalid_plus_context_mass") for row in rows)
    vals_stable = finite(row.get("read_stable_mass") for row in rows)
    hi = quantile(vals_invalid, 0.75)
    lo = quantile(vals_stable, 0.25)
    if hi is not None and lo is not None:
        candidates.append(
            {
                "cue_id": "READ_UNSTABLE_GE_Q75_AND_STABLE_LE_Q25",
                "feature": "read_invalid_plus_context_mass",
                "direction": "ge",
                "threshold": hi,
                "feature2": "read_stable_mass",
                "direction2": "le",
                "threshold2": lo,
            }
        )
    return candidates[:max_candidates]


def selected_mask(rows: list[Mapping[str, Any]], candidate: Mapping[str, Any]) -> list[bool]:
    feature = str(candidate["feature"])
    direction = str(candidate["direction"])
    threshold = f(candidate["threshold"])
    mask = []
    for row in rows:
        value = f(row.get(feature))
        hit = value >= threshold if direction == "ge" else value <= threshold
        if candidate.get("feature2"):
            value2 = f(row.get(candidate["feature2"]))
            threshold2 = f(candidate["threshold2"])
            direction2 = str(candidate["direction2"])
            hit2 = value2 >= threshold2 if direction2 == "ge" else value2 <= threshold2
            hit = hit and hit2
        mask.append(bool(hit and math.isfinite(value)))
    return mask


def balanced_metrics(rows: list[Mapping[str, Any]], mask: list[bool], include_ids: bool = True) -> dict[str, Any]:
    positives = [idx for idx, row in enumerate(rows) if row.get("v95_case_bucket") == "READ_LOCAL_BAD"]
    negatives = [idx for idx, row in enumerate(rows) if row.get("case_label_offline_only") == "good"]
    pos_hits = [idx for idx in positives if mask[idx]]
    neg_hits = [idx for idx in negatives if mask[idx]]
    bad_recall = len(pos_hits) / max(len(positives), 1)
    good_fpr = len(neg_hits) / max(len(negatives), 1)
    out: dict[str, Any] = {
        "selected_count": int(sum(mask)),
        "positive_total": len(positives),
        "negative_total": len(negatives),
        "selected_positive_count": len(pos_hits),
        "selected_negative_count": len(neg_hits),
        "bad_recall": bad_recall,
        "good_FPR": good_fpr,
        "balanced_accuracy": 0.5 * (bad_recall + (1.0 - good_fpr)),
        "positive_sequence_coverage": len({str(rows[idx].get("seq")) for idx in pos_hits}),
        "selected_sequence_coverage": len({str(row.get("seq")) for idx, row in enumerate(rows) if mask[idx]}),
    }
    if include_ids:
        out.update(
            {
                "selected_pair_ids": ",".join(str(row.get("pair_id")) for idx, row in enumerate(rows) if mask[idx]),
                "selected_positive_pair_ids": ",".join(str(rows[idx].get("pair_id")) for idx in pos_hits),
                "selected_negative_pair_ids": ",".join(str(rows[idx].get("pair_id")) for idx in neg_hits),
            }
        )
    return out


def evaluate_candidate(rows: list[Mapping[str, Any]], candidate: Mapping[str, Any], random_seeds: int) -> dict[str, Any]:
    mask = selected_mask(rows, candidate)
    actual = balanced_metrics(rows, mask)
    selected_count = int(actual["selected_count"])
    global_random = [
        balanced_metrics(rows, stable_random_mask(rows, selected_count, seed), include_ids=False)["balanced_accuracy"]
        for seed in range(random_seeds)
    ]
    seq_random = [
        balanced_metrics(rows, seq_count_random_mask(rows, mask, seed), include_ids=False)["balanced_accuracy"]
        for seed in range(random_seeds)
    ]
    global_p95 = quantile(global_random, 0.95)
    seq_p95 = quantile(seq_random, 0.95)
    gates = {
        "bad_recall_gate": actual["bad_recall"] >= 0.60,
        "good_FPR_gate": actual["good_FPR"] <= 0.25,
        "positive_sequence_coverage_gate": actual["positive_sequence_coverage"] >= 3,
        "global_same_count_margin_gate": global_p95 is not None and actual["balanced_accuracy"] > global_p95,
        "seq_count_margin_gate": seq_p95 is not None and actual["balanced_accuracy"] > seq_p95,
    }
    return {
        **candidate,
        **actual,
        "global_same_count_random_ba_p95": global_p95,
        "seq_count_random_ba_p95": seq_p95,
        "global_same_count_margin": None if global_p95 is None else actual["balanced_accuracy"] - global_p95,
        "seq_count_margin": None if seq_p95 is None else actual["balanced_accuracy"] - seq_p95,
        **gates,
        "candidate_gate_pass": all(gates.values()),
    }


def rank_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        bool(row.get("candidate_gate_pass")),
        f(row.get("bad_recall"), -1.0),
        -f(row.get("good_FPR"), 2.0),
        f(row.get("balanced_accuracy"), -1.0),
        f(row.get("global_same_count_margin"), -999.0),
    )


def main() -> None:
    args = parse_args()
    read_rows = read_csv(args.read_rows)
    case_rows = read_csv(args.case_rows)
    rows = join_rows(read_rows, case_rows)
    add_derived_scores(rows)
    candidates = build_candidates(rows, args.max_candidates)
    metrics = [evaluate_candidate(rows, candidate, args.random_seeds) for candidate in candidates]
    metrics.sort(key=rank_key, reverse=True)
    passing = [row for row in metrics if row.get("candidate_gate_pass")]
    best = metrics[0] if metrics else {}
    qk_available = sum(1 for row in rows if str(row.get("read_QK_compatibility") or "").strip())
    summary = {
        "phase": "v95_trackD_read_cue_validation_v1",
        "read_rows": str(args.read_rows),
        "case_rows": str(args.case_rows),
        "row_count": len(rows),
        "candidate_count": len(metrics),
        "candidate_passing_count": len(passing),
        "gate_pass": bool(passing),
        "runtime_action_allowed": False,
        "read_local_positive_count": sum(1 for row in rows if row.get("v95_case_bucket") == "READ_LOCAL_BAD"),
        "good_control_count": sum(1 for row in rows if row.get("case_label_offline_only") == "good"),
        "qk_compatibility_available_rows": qk_available,
        "best_candidate": best,
        "blocker": (
            ""
            if passing and qk_available
            else "read_QK_compatibility_unavailable"
            if passing
            else "no_read_cue_candidate_passes_bad_good_random_controls;read_QK_compatibility_unavailable"
        ),
        "interpretation_boundary": (
            "This validates existing READ cue mass rows only. It does not prove a READ action surface or QK compatibility."
        ),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "read_cue_candidate_metrics.csv", metrics)
    write_csv(args.out_dir / "read_cue_joined_rows.csv", rows)
    write_json(args.out_dir / "summary.json", summary)
    write_text(
        args.out_dir / "analysis.md",
        f"""
# Track D READ Cue Validation

- row_count: `{summary['row_count']}`
- read_local_positive_count: `{summary['read_local_positive_count']}`
- good_control_count: `{summary['good_control_count']}`
- candidate_count: `{summary['candidate_count']}`
- candidate_passing_count: `{summary['candidate_passing_count']}`
- qk_compatibility_available_rows: `{summary['qk_compatibility_available_rows']}`
- best_candidate: `{best.get('cue_id')}`
- best_bad_recall: `{best.get('bad_recall')}`
- best_good_FPR: `{best.get('good_FPR')}`
- best_balanced_accuracy: `{best.get('balanced_accuracy')}`
- best_global_same_count_margin: `{best.get('global_same_count_margin')}`
- best_seq_count_margin: `{best.get('seq_count_margin')}`
- gate_pass: `{summary['gate_pass']}`
- runtime_action_allowed: `{summary['runtime_action_allowed']}`
- blocker: `{summary['blocker']}`

Interpretation: this is a diagnostic-only bad/good control check for existing
READ cue mass rows. A passing cue would still require QK compatibility and a
measured READ action surface before runtime promotion.
""",
    )
    write_text(
        args.out_dir / "what_would_have_to_be_true_to_pass.md",
        """
To pass Track D cue validation, a method-safe READ cue must reach bad_recall >= 0.60,
good_FPR <= 0.25, positive sequence coverage >= 3, and beat global/sequence
same-count random controls. To promote beyond cue validation, READ QK compatibility
and a measured READ-local L2 action surface are still required.
""",
    )
    print(f"row_count={summary['row_count']}")
    print(f"read_local_positive_count={summary['read_local_positive_count']}")
    print(f"good_control_count={summary['good_control_count']}")
    print(f"candidate_count={summary['candidate_count']}")
    print(f"candidate_passing_count={summary['candidate_passing_count']}")
    print(f"qk_compatibility_available_rows={summary['qk_compatibility_available_rows']}")
    print(f"best_candidate={best.get('cue_id')}")
    print(f"best_bad_recall={best.get('bad_recall')}")
    print(f"best_good_FPR={best.get('good_FPR')}")
    print(f"best_balanced_accuracy={best.get('balanced_accuracy')}")
    print(f"best_global_same_count_margin={best.get('global_same_count_margin')}")
    print(f"best_seq_count_margin={best.get('seq_count_margin')}")
    print(f"gate_pass={summary['gate_pass']}")
    print(f"runtime_action_allowed={summary['runtime_action_allowed']}")
    print(f"blocker={summary['blocker']}")


if __name__ == "__main__":
    main()
