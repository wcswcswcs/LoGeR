#!/usr/bin/env python3
"""Audit a v95 Track G SWA handoff cue bank against labelled handoff cases.

This is a diagnostic/cue-bank audit, not a runtime action. It evaluates
training-free cue atoms on the v95 Track A SWA handoff labelled universe:
positive rows are labelled bad SWA_HANDOFF cases; negative rows are labelled
good controls. Unlabelled support rows are reported for coverage only.
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

import pandas as pd


ROOT = Path("results/acl2_v95tf_multiroute_semantic_memory_evidence_control")
V94_ROOT = Path("results/acl2_v94tf_semantic_gauge_failure_localization_causal_memory_control")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metric-rows", type=Path, default=ROOT / "metric_suite/metric_rows.csv")
    parser.add_argument("--swa-cases", type=Path, default=ROOT / "trackA_base_case_bank/swa_handoff_cases.csv")
    parser.add_argument("--good-controls", type=Path, default=ROOT / "trackA_base_case_bank/good_controls.csv")
    parser.add_argument("--selected-rows", type=Path, default=V94_ROOT / "phase5_object_source_extension/selected_policy_rows.csv")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "trackG_swa_handoff_cue_bank_v2")
    parser.add_argument("--random-seeds", type=int, default=32)
    parser.add_argument("--max-method-control-candidates", type=int, default=80)
    parser.add_argument("--max-diagnostic-control-candidates", type=int, default=40)
    return parser.parse_args()


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


def q(values: pd.Series, quantile: float) -> float:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    return float(numeric.quantile(quantile)) if len(numeric) else float("nan")


def stable_unit(*parts: Any) -> float:
    digest = hashlib.sha256("|".join(str(part) for part in parts).encode("utf-8")).hexdigest()
    return int(digest[:12], 16) / float(16**12)


def rotate(values: list[bool], amount: int) -> list[bool]:
    if not values:
        return []
    amount %= len(values)
    return values[amount:] + values[:amount]


def bool_text(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def and_mask(*masks: list[bool]) -> list[bool]:
    return [all(values) for values in zip(*masks)]


def or_mask(*masks: list[bool]) -> list[bool]:
    return [any(values) for values in zip(*masks)]


def normalize_metric_rows(metric: pd.DataFrame) -> pd.DataFrame:
    out = metric.copy()
    if "case_id" in out.columns and "pair_id" not in out.columns:
        out = out.rename(columns={"case_id": "pair_id"})
    out["pair_id"] = out["pair_id"].astype(str)
    return out


def build_eval_rows(metric: pd.DataFrame, swa: pd.DataFrame, good: pd.DataFrame, selected: pd.DataFrame) -> pd.DataFrame:
    metric = normalize_metric_rows(metric)
    selected = selected.copy()
    selected["pair_id"] = selected["pair_id"].astype(str)
    swa_ids = set(swa["case_id"].astype(str))
    good_ids = set(good["case_id"].astype(str))
    rows: list[dict[str, Any]] = []
    for row in metric.to_dict(orient="records"):
        pair_id = str(row.get("pair_id", ""))
        label = str(row.get("case_label_offline_only", ""))
        if pair_id in swa_ids and label == "bad":
            out = dict(row)
            out["cue_eval_label"] = "positive"
            out["cue_eval_role"] = "labelled_bad_swa_handoff"
            rows.append(out)
        elif pair_id in good_ids and label == "good":
            out = dict(row)
            out["cue_eval_label"] = "negative"
            out["cue_eval_role"] = "labelled_good_control"
            rows.append(out)
    eval_df = pd.DataFrame(rows)
    if eval_df.empty:
        return eval_df
    return eval_df.merge(selected, on="pair_id", how="left", suffixes=("", "_selected"))


def atom_thresholds(frame: pd.DataFrame, columns: list[str]) -> tuple[dict[str, list[bool]], dict[str, float]]:
    atoms: dict[str, list[bool]] = {}
    thresholds: dict[str, float] = {}
    for column in columns:
        if column not in frame.columns:
            continue
        values = pd.to_numeric(frame[column], errors="coerce")
        if not values.notna().any():
            continue
        for quantile in [0.25, 0.50, 0.60, 0.75, 0.85]:
            suffix = int(quantile * 100)
            threshold = q(values, quantile)
            thresholds[f"{column}_q{suffix}"] = threshold
            atoms[f"{column.upper()}_GE_Q{suffix}"] = (values >= threshold).fillna(False).tolist()
            atoms[f"{column.upper()}_LE_Q{suffix}"] = (values <= threshold).fillna(False).tolist()
    return atoms, thresholds


def build_atoms(frame: pd.DataFrame) -> tuple[dict[str, list[bool]], dict[str, list[bool]], dict[str, float]]:
    method_columns = [
        "semantic_invalid_mass",
        "semantic_context_mass",
        "semantic_dynamic_region_mass",
        "semantic_object_boundary_mass",
        "semantic_low_observability_score",
        "semantic_multimode_conflict_score",
        "object_boundary_ratio",
        "boundary_global_cross_ratio",
        "boundary_new_id_ratio",
        "radio_boundary_mean",
    ]
    diagnostic_columns = [
        "L1_local_sim3_ate",
        "L2_intra_scale_cv",
        "L2_head_tail_proxy_error",
        "L3_adjacent_log_scale_jump",
        "L3_handoff_transfer_penalty_proxy",
        "L3_gauge_jump_proxy",
        "L3_J_handoff",
        "carrier_error_merge_residual_after_abs",
        "carrier_error_abs_log_scale_jump_runtime",
    ]
    method_atoms, thresholds = atom_thresholds(frame, method_columns)
    diagnostic_atoms, diagnostic_thresholds = atom_thresholds(frame, diagnostic_columns)
    thresholds.update(diagnostic_thresholds)

    for column in ["semantic_evidence_type", "semantic_evidence_type_selected", "selected_policy"]:
        if column in frame.columns:
            for value in sorted(frame[column].dropna().astype(str).unique()):
                safe = value.replace(" ", "_").replace("/", "_")
                method_atoms[f"{column.upper()}_EQ_{safe}"] = (frame[column].astype(str) == value).tolist()
    if "selected_policy_positive" in frame.columns:
        method_atoms["SELECTED_POLICY_POSITIVE"] = [bool_text(v) for v in frame["selected_policy_positive"]]

    diagnostic_atoms.update(method_atoms)
    if "failure_type_primary" in frame.columns:
        for value in sorted(frame["failure_type_primary"].dropna().astype(str).unique()):
            safe = value.replace(" ", "_").replace("/", "_")
            diagnostic_atoms[f"FAILURE_TYPE_PRIMARY_EQ_{safe}"] = (frame["failure_type_primary"].astype(str) == value).tolist()
    return method_atoms, diagnostic_atoms, thresholds


def candidate_masks(atoms: dict[str, list[bool]], include_pairs: bool = True) -> dict[str, list[bool]]:
    out: dict[str, list[bool]] = {name: mask for name, mask in atoms.items() if any(mask)}
    if not include_pairs:
        return out
    names = sorted(out)
    for left, right in combinations(names, 2):
        out[f"{left}__AND__{right}"] = and_mask(out[left], out[right])
    return {name: mask for name, mask in out.items() if any(mask)}


def balanced_metrics(frame: pd.DataFrame, mask: list[bool], include_hits: bool = True) -> dict[str, Any]:
    labels = frame["cue_eval_label"].astype(str).tolist()
    positives = [label == "positive" for label in labels]
    negatives = [label == "negative" for label in labels]
    selected = [idx for idx, value in enumerate(mask) if value]
    tp = sum(mask[idx] and positives[idx] for idx in range(len(mask)))
    fp = sum(mask[idx] and negatives[idx] for idx in range(len(mask)))
    positive_total = sum(positives)
    negative_total = sum(negatives)
    recall = tp / max(positive_total, 1)
    fpr = fp / max(negative_total, 1)
    balanced_accuracy = (recall + (1.0 - fpr)) / 2.0
    out = {
        "selected_count": len(selected),
        "selected_positive_count": tp,
        "selected_negative_count": fp,
        "positive_total": positive_total,
        "negative_total": negative_total,
        "bad_recall": recall,
        "good_FPR": fpr,
        "balanced_accuracy": balanced_accuracy,
        "positive_sequence_coverage": len(set(frame.iloc[[idx for idx in selected if positives[idx]]]["seq"].astype(str).tolist())),
        "selected_sequence_coverage": len(set(frame.iloc[selected]["seq"].astype(str).tolist())) if selected else 0,
    }
    if include_hits:
        out.update(
            {
                "selected_pair_ids": ",".join(frame.iloc[selected]["pair_id"].astype(str).tolist()),
                "selected_positive_pair_ids": ",".join(
                    frame.iloc[[idx for idx in selected if positives[idx]]]["pair_id"].astype(str).tolist()
                ),
                "selected_negative_pair_ids": ",".join(
                    frame.iloc[[idx for idx in selected if negatives[idx]]]["pair_id"].astype(str).tolist()
                ),
            }
        )
    return out


def random_mask(frame: pd.DataFrame, count: int, seed: int) -> list[bool]:
    ordered = sorted(range(len(frame)), key=lambda idx: stable_unit("global", seed, frame.iloc[idx]["pair_id"]))
    selected = set(ordered[: min(count, len(ordered))])
    return [idx in selected for idx in range(len(frame))]


def seq_count_random_mask(frame: pd.DataFrame, mask: list[bool], seed: int) -> list[bool]:
    selected: set[int] = set()
    for seq, group in frame.groupby(frame["seq"].astype(str), sort=False):
        indices = list(group.index)
        count = sum(mask[idx] for idx in indices)
        ordered = sorted(indices, key=lambda idx: stable_unit("seq", seed, seq, frame.loc[idx, "pair_id"]))
        selected.update(ordered[: min(count, len(ordered))])
    return [idx in selected for idx in range(len(frame))]


def semantic_rotation_masks(frame: pd.DataFrame, mask: list[bool]) -> list[list[bool]]:
    sort_cols = [
        col
        for col in [
            "semantic_evidence_type",
            "selected_policy",
            "semantic_low_observability_score",
            "semantic_object_boundary_mass",
            "semantic_multimode_conflict_score",
            "pair_id",
        ]
        if col in frame.columns
    ]
    ordered = list(frame.sort_values(sort_cols).index) if sort_cols else list(frame.index)
    values = [mask[idx] for idx in ordered]
    masks: list[list[bool]] = []
    for amount in range(1, len(values)):
        rotated = rotate(values, amount)
        out = [False] * len(mask)
        for idx, value in zip(ordered, rotated):
            out[idx] = value
        masks.append(out)
    return masks


def percentile(values: list[float], pct: float) -> float | None:
    finite = sorted(v for v in values if math.isfinite(v))
    if not finite:
        return None
    pos = (len(finite) - 1) * pct
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(finite[lo])
    return float(finite[lo] * (hi - pos) + finite[hi] * (pos - lo))


def evaluate(frame: pd.DataFrame, mask: list[bool], name: str, scope: str, random_seeds: int) -> dict[str, Any]:
    actual = balanced_metrics(frame, mask, include_hits=True)
    selected_count = int(actual["selected_count"])
    global_bas = [
        balanced_metrics(frame, random_mask(frame, selected_count, seed), include_hits=False)["balanced_accuracy"]
        for seed in range(random_seeds)
    ]
    seq_bas = [
        balanced_metrics(frame, seq_count_random_mask(frame, mask, seed), include_hits=False)["balanced_accuracy"]
        for seed in range(random_seeds)
    ]
    sem_bas = [
        balanced_metrics(frame, control, include_hits=False)["balanced_accuracy"]
        for control in semantic_rotation_masks(frame, mask)
    ]
    global_p95 = percentile(global_bas, 0.95)
    seq_p95 = percentile(seq_bas, 0.95)
    sem_p95 = percentile(sem_bas, 0.95)
    actual_ba = float(actual["balanced_accuracy"])
    out = {
        "cue_id": name,
        "scope": scope,
        **actual,
        "global_same_count_random_ba_p95": global_p95,
        "seq_count_random_ba_p95": seq_p95,
        "semantic_rotation_ba_p95": sem_p95,
        "global_same_count_margin": actual_ba - global_p95 if global_p95 is not None else None,
        "seq_count_margin": actual_ba - seq_p95 if seq_p95 is not None else None,
        "semantic_rotation_margin": actual_ba - sem_p95 if sem_p95 is not None else None,
    }
    gates = {
        "bad_recall_gate": out["bad_recall"] >= 0.60,
        "good_FPR_gate": out["good_FPR"] <= 0.25,
        "positive_sequence_coverage_gate": int(out["positive_sequence_coverage"]) >= 2,
        "global_same_count_margin_gate": out["global_same_count_margin"] is not None and out["global_same_count_margin"] >= 0.05,
        "seq_count_margin_gate": out["seq_count_margin"] is not None and out["seq_count_margin"] >= 0.05,
        "semantic_rotation_margin_gate": out["semantic_rotation_margin"] is not None and out["semantic_rotation_margin"] >= 0.05,
    }
    out.update(gates)
    out["candidate_gate_pass"] = all(gates.values())
    return out


def basic_evaluate(frame: pd.DataFrame, mask: list[bool], name: str, scope: str) -> dict[str, Any]:
    actual = balanced_metrics(frame, mask, include_hits=False)
    out = {
        "cue_id": name,
        "scope": scope,
        **actual,
        "control_evaluated": False,
        "global_same_count_random_ba_p95": None,
        "seq_count_random_ba_p95": None,
        "semantic_rotation_ba_p95": None,
        "global_same_count_margin": None,
        "seq_count_margin": None,
        "semantic_rotation_margin": None,
    }
    out.update(
        {
            "bad_recall_gate": out["bad_recall"] >= 0.60,
            "good_FPR_gate": out["good_FPR"] <= 0.25,
            "positive_sequence_coverage_gate": int(out["positive_sequence_coverage"]) >= 2,
            "global_same_count_margin_gate": False,
            "seq_count_margin_gate": False,
            "semantic_rotation_margin_gate": False,
            "candidate_gate_pass": False,
        }
    )
    return out


def basic_rank_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        bool_text(row.get("bad_recall_gate")) and bool_text(row.get("good_FPR_gate")) and bool_text(row.get("positive_sequence_coverage_gate")),
        f(row.get("balanced_accuracy"), -1.0),
        f(row.get("bad_recall"), -1.0),
        -f(row.get("good_FPR"), 999.0),
        int(row.get("positive_sequence_coverage") or 0),
        int(row.get("selected_count") or 0),
    )


def select_control_candidates(
    basics: list[dict[str, Any]],
    candidates: dict[str, list[bool]],
    limit: int,
) -> list[tuple[str, list[bool]]]:
    ranked = sorted(basics, key=basic_rank_key, reverse=True)
    selected: list[tuple[str, list[bool]]] = []
    seen: set[str] = set()
    for row in ranked:
        if not (
            bool_text(row.get("bad_recall_gate"))
            and bool_text(row.get("good_FPR_gate"))
            and bool_text(row.get("positive_sequence_coverage_gate"))
        ):
            continue
        name = str(row["cue_id"])
        if name in candidates and name not in seen:
            selected.append((name, candidates[name]))
            seen.add(name)
        if len(selected) >= limit:
            return selected
    for row in ranked:
        name = str(row["cue_id"])
        if name in candidates and name not in seen:
            selected.append((name, candidates[name]))
            seen.add(name)
        if len(selected) >= limit:
            break
    return selected


def rank_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        bool_text(row.get("candidate_gate_pass")),
        f(row.get("balanced_accuracy"), -1.0),
        f(row.get("semantic_rotation_margin"), -999.0),
        f(row.get("seq_count_margin"), -999.0),
        f(row.get("bad_recall"), -1.0),
        -f(row.get("good_FPR"), 999.0),
    )


def main() -> None:
    args = parse_args()
    metric = pd.read_csv(args.metric_rows)
    swa = pd.read_csv(args.swa_cases)
    good = pd.read_csv(args.good_controls)
    selected = pd.read_csv(args.selected_rows)
    eval_df = build_eval_rows(metric, swa, good, selected)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    eval_rows = eval_df.to_dict(orient="records")
    write_csv(args.out_dir / "evaluation_rows.csv", eval_rows)

    if eval_df.empty:
        summary = {"gate_pass": False, "blocker": "empty_evaluation_universe"}
        write_json(args.out_dir / "summary.json", summary)
        print("blocker=empty_evaluation_universe")
        return

    method_atoms, diagnostic_atoms, thresholds = build_atoms(eval_df)
    method_candidates = candidate_masks(method_atoms, include_pairs=True)
    diagnostic_candidates = candidate_masks(diagnostic_atoms, include_pairs=False)

    method_basic = [
        basic_evaluate(eval_df, mask, name, "method_safe_no_gt_semantic_source") for name, mask in method_candidates.items()
    ]
    diagnostic_basic = [
        basic_evaluate(eval_df, mask, name, "diagnostic_only_includes_offline_metrics") for name, mask in diagnostic_candidates.items()
    ]
    method_for_controls = select_control_candidates(method_basic, method_candidates, args.max_method_control_candidates)
    diagnostic_for_controls = select_control_candidates(
        diagnostic_basic, diagnostic_candidates, args.max_diagnostic_control_candidates
    )
    method_metrics = [
        evaluate(eval_df, mask, name, "method_safe_no_gt_semantic_source", args.random_seeds)
        for name, mask in method_for_controls
    ]
    diagnostic_metrics = [
        evaluate(eval_df, mask, name, "diagnostic_only_includes_offline_metrics", args.random_seeds)
        for name, mask in diagnostic_for_controls
    ]
    method_metrics.sort(key=rank_key, reverse=True)
    diagnostic_metrics.sort(key=rank_key, reverse=True)

    method_basic.sort(key=basic_rank_key, reverse=True)
    diagnostic_basic.sort(key=basic_rank_key, reverse=True)
    write_csv(args.out_dir / "method_safe_candidate_basic_metrics.csv", method_basic)
    write_csv(args.out_dir / "diagnostic_only_candidate_basic_metrics.csv", diagnostic_basic)
    write_csv(args.out_dir / "method_safe_candidate_metrics.csv", method_metrics)
    write_csv(args.out_dir / "diagnostic_only_candidate_metrics.csv", diagnostic_metrics)
    write_csv(args.out_dir / "candidate_cue_metrics.csv", method_metrics + diagnostic_metrics)
    write_json(args.out_dir / "thresholds.json", thresholds)

    method_passing = [row for row in method_metrics if bool_text(row.get("candidate_gate_pass"))]
    diagnostic_passing = [row for row in diagnostic_metrics if bool_text(row.get("candidate_gate_pass"))]
    unlabelled_swa_count = int((swa["case_label_offline_only"].astype(str) == "unlabelled_support").sum())
    summary = {
        "stage": "TrackG_G5_SWA_handoff_cue_bank_v2",
        "labelled_universe_count": int(len(eval_df)),
        "positive_count": int((eval_df["cue_eval_label"] == "positive").sum()),
        "negative_count": int((eval_df["cue_eval_label"] == "negative").sum()),
        "unlabelled_swa_support_count": unlabelled_swa_count,
        "method_candidate_count": int(len(method_metrics)),
        "diagnostic_candidate_count": int(len(diagnostic_metrics)),
        "method_candidate_basic_count": int(len(method_basic)),
        "diagnostic_candidate_basic_count": int(len(diagnostic_basic)),
        "method_control_evaluated_count": int(len(method_metrics)),
        "diagnostic_control_evaluated_count": int(len(diagnostic_metrics)),
        "method_safe_passing_count": int(len(method_passing)),
        "diagnostic_only_passing_count": int(len(diagnostic_passing)),
        "best_method_safe": method_metrics[0] if method_metrics else {},
        "best_diagnostic_only": diagnostic_metrics[0] if diagnostic_metrics else {},
        "gate_pass": bool(method_passing),
        "runtime_action_allowed": False,
        "blocker": "" if method_passing else "no_method_safe_swa_cue_passes_g5_controls",
        "action_note": "cue service readiness does not override Track E action-surface strict handoff failure",
    }
    write_json(args.out_dir / "summary.json", summary)
    best = summary["best_method_safe"]
    write_text(
        args.out_dir / "analysis.md",
        f"""
# Track G SWA Handoff Cue Bank v2

- Labelled universe: `{summary['labelled_universe_count']}` rows (`{summary['positive_count']}` positive bad SWA handoff, `{summary['negative_count']}` good controls).
- Unlabelled SWA support rows excluded from recall/FPR: `{summary['unlabelled_swa_support_count']}`.
- Method-safe basic candidate count: `{summary['method_candidate_basic_count']}`; control-evaluated: `{summary['method_control_evaluated_count']}`; passing: `{summary['method_safe_passing_count']}`.
- Diagnostic-only basic candidate count: `{summary['diagnostic_candidate_basic_count']}`; control-evaluated: `{summary['diagnostic_control_evaluated_count']}`; passing: `{summary['diagnostic_only_passing_count']}`.
- Best method-safe cue: `{best.get('cue_id')}`.
- Best method-safe bad_recall/good_FPR/BA: `{best.get('bad_recall')}` / `{best.get('good_FPR')}` / `{best.get('balanced_accuracy')}`.
- Best method-safe control margins global/seq/semantic-rotation: `{best.get('global_same_count_margin')}` / `{best.get('seq_count_margin')}` / `{best.get('semantic_rotation_margin')}`.

This audit only evaluates cue specificity. Runtime action remains blocked until a Track E mechanism improves strict handoff metrics and protects good controls.
""",
    )
    recommendation = (
        "trackG_swa_cue_service_ready_but_trackE_action_mechanism_blocked"
        if method_passing
        else "continue_trackG_cue_mining_before_trackE_action"
    )
    write_text(args.out_dir / "next_route_recommendation.md", recommendation)
    print(f"labelled_universe_count={summary['labelled_universe_count']}")
    print(f"positive_count={summary['positive_count']}")
    print(f"negative_count={summary['negative_count']}")
    print(f"method_candidate_basic_count={summary['method_candidate_basic_count']}")
    print(f"method_control_evaluated_count={summary['method_control_evaluated_count']}")
    print(f"method_safe_passing_count={summary['method_safe_passing_count']}")
    print(f"diagnostic_candidate_basic_count={summary['diagnostic_candidate_basic_count']}")
    print(f"diagnostic_control_evaluated_count={summary['diagnostic_control_evaluated_count']}")
    print(f"diagnostic_only_passing_count={summary['diagnostic_only_passing_count']}")
    print(f"best_method_safe={best.get('cue_id')}")
    print(f"best_method_safe_bad_recall={best.get('bad_recall')}")
    print(f"best_method_safe_good_FPR={best.get('good_FPR')}")
    print(f"best_method_safe_semantic_rotation_margin={best.get('semantic_rotation_margin')}")
    print(f"gate_pass={summary['gate_pass']}")
    print(f"next_route_recommendation={recommendation}")


if __name__ == "__main__":
    main()
