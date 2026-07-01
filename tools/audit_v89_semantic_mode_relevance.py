#!/usr/bin/env python3
"""Audit v89 Phase2 semantic mode relevance against geometry-only controls."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from v86_soft_latent_utils import spearman_rho, stable_hash_float, write_csv, write_json


DEFAULT_ROOT = Path("results/acl2_v89tf_semantic_scale_mode_observability_memory_control")
DEFAULT_LEDGER = DEFAULT_ROOT / "phase1_semantic_scale_mode_ledger"
DEFAULT_V88_REL = Path("results/acl2_v88tf_scale_mode_consensus_gauge_update_attribution/phase2_scale_mode_relevance/scale_mode_relevance_rows.csv")
DEFAULT_OUT = DEFAULT_ROOT / "phase2_semantic_mode_relevance"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger-dir", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--v88-relevance-rows", type=Path, default=DEFAULT_V88_REL)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--filter", choices=["all", "highobs", "nonseq01", "near", "far", "semantic_structure_rich", "semantic_lowobs"], default="all")
    return parser.parse_args()


def _num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _shuffle(values: pd.Series, salt: str) -> pd.Series:
    arr = values.to_numpy(copy=True)
    out = arr.copy()
    if len(arr) <= 1:
        return pd.Series(out, index=values.index)
    order = sorted(range(len(arr)), key=lambda i: stable_hash_float(salt, i))
    shuffled = arr[order]
    shuffled = np.roll(shuffled, 1)
    for dst, value in zip(order, shuffled):
        out[dst] = value
    return pd.Series(out, index=values.index)


def _load_rows(ledger_dir: Path, v88_rel: Path) -> pd.DataFrame:
    df = pd.read_csv(ledger_dir / "semantic_scale_pair_rows.csv")
    df["seq"] = df["seq"].astype(str).str.zfill(2)
    df["prev_chunk"] = df["prev_chunk"].astype(int)
    df["curr_chunk"] = df["curr_chunk"].astype(int)
    if v88_rel.exists():
        rel = pd.read_csv(v88_rel)
        rel["seq"] = rel["seq"].astype(str).str.zfill(2)
        rel["prev_chunk"] = rel["prev_chunk"].astype(int)
        rel["curr_chunk"] = rel["curr_chunk"].astype(int)
        keep = [
            "seq",
            "prev_chunk",
            "curr_chunk",
            "M0_v87_S_overlap_baseline",
            "M3_mode_mad",
            "M4_mode_entropy",
            "M7_native_mode_sign_mismatch",
            "M12_observability_only",
        ]
        df = df.merge(rel[keep], on=["seq", "prev_chunk", "curr_chunk"], how="left")
    for col in ["M0_v87_S_overlap_baseline", "M3_mode_mad", "M4_mode_entropy", "M7_native_mode_sign_mismatch", "M12_observability_only"]:
        if col not in df:
            df[col] = np.nan
    return df


def _apply_filter(df: pd.DataFrame, name: str) -> tuple[pd.DataFrame, list[str]]:
    notes: list[str] = []
    out = df.copy()
    if name == "highobs":
        q = _num(out["observability_score"]).quantile(0.50)
        before = len(out)
        out = out[_num(out["observability_score"]) >= q].copy()
        notes.append(f"highobs:q50={q}:{before}->{len(out)}")
    elif name == "nonseq01":
        before = len(out)
        out = out[out["seq"] != "01"].copy()
        notes.append(f"nonseq01:{before}->{len(out)}")
    elif name in {"near", "far"}:
        q = _num(out["geometry_dominant_mode_mu"]).abs().quantile(0.50)
        before = len(out)
        mask = _num(out["geometry_dominant_mode_mu"]).abs() <= q
        out = out[mask if name == "near" else ~mask].copy()
        notes.append(f"{name}_abs_geometry_mode_split:q50={q}:{before}->{len(out)}")
    elif name == "semantic_structure_rich":
        q = _num(out["semantic_valid_mass"]).quantile(0.75)
        before = len(out)
        out = out[_num(out["semantic_valid_mass"]) >= q].copy()
        notes.append(f"semantic_structure_rich:q75_valid_mass={q}:{before}->{len(out)}")
    elif name == "semantic_lowobs":
        q = _num(out["semantic_lowobs_mass"]).quantile(0.75)
        before = len(out)
        out = out[_num(out["semantic_lowobs_mass"]) >= q].copy()
        notes.append(f"semantic_lowobs:q75_lowobs_mass={q}:{before}->{len(out)}")
    return out, notes


def _scores(df: pd.DataFrame) -> dict[str, pd.Series]:
    geom = _num(df["geometry_dominant_mode_mu"]).abs().fillna(0.0)
    sem_valid_mu = _num(df["semantic_valid_dominant_mode_mu"]).abs().fillna(0.0)
    sem_invalid_mu = _num(df["semantic_invalid_dominant_mode_mu"]).abs().fillna(0.0)
    valid_mass = _num(df["semantic_valid_mass"]).fillna(0.0)
    invalid_mass = _num(df["semantic_invalid_mass"]).fillna(0.0)
    context = _num(df["semantic_context_mass"]).fillna(0.0)
    lowobs = _num(df["semantic_lowobs_mass"]).fillna(0.0)
    entropy_red = _num(df["semantic_entropy_reduction"]).fillna(0.0)
    osem = _num(df["O_sem_scale"]).fillna(0.0)
    native = _num(df["native_delta_log_scale"]).fillna(0.0)
    return {
        "G0_geometry_dominant_mode": geom,
        "G1_v88_overlap_baseline": _num(df["M0_v88_S_overlap_baseline"] if "M0_v88_S_overlap_baseline" in df else df["M0_v87_S_overlap_baseline"]).fillna(0.0),
        "G2_mode_entropy_mad": (_num(df["M3_mode_mad"]).fillna(0.0) + _num(df["M4_mode_entropy"]).fillna(0.0)) / 2.0,
        "S0_semantic_valid_dominant_mode": sem_valid_mu * (1.0 + valid_mass),
        "S1_semantic_invalid_conflict": sem_invalid_mu * invalid_mass,
        "S2_semantic_context_lowobs": context + lowobs,
        "S3_semantic_entropy_reduction": entropy_red.clip(lower=0.0),
        "S4_semantic_observability_O_sem_scale": osem,
        "S5_semantic_valid_native_mismatch": (native - _num(df["semantic_valid_dominant_mode_mu"]).fillna(0.0)).abs() * (1.0 + valid_mass),
        "S6_semantic_mode_type_combined": (sem_invalid_mu * invalid_mass) + (context + lowobs) - (0.5 * valid_mass),
        "C0_geometry_plus_semantic_valid": geom * (1.0 + valid_mass),
        "C1_geometry_plus_semantic_veto": geom * (1.0 + invalid_mass + context + lowobs),
        "C2_geometry_plus_semantic_observability": geom * (1.0 + osem),
    }


def _metric(df: pd.DataFrame, signal: str, values: pd.Series, geometry_ref: dict[str, Any]) -> dict[str, Any]:
    labelled = df[_num(df["abs_log_scale_jump_gt"]).notna()].copy()
    v = values.loc[labelled.index]
    y = _num(labelled["abs_log_scale_jump_gt"])
    scale_threshold = float(y.quantile(0.75)) if len(y) else float("nan")
    threshold = float(v.quantile(0.75)) if len(v) else float("nan")
    flags = v >= threshold
    high = y >= scale_threshold
    bad = labelled["base_case_type"].astype(str).eq("bad")
    good_low = labelled["base_case_type"].astype(str).eq("good") & (~high)
    good = labelled["base_case_type"].astype(str).eq("good")
    rho = spearman_rho(v.tolist(), y.tolist())
    controls = {
        "label_shuffle": _shuffle(v, f"{signal}:label_shuffle"),
        "confidence_shuffle": _shuffle(v * (_num(labelled.get("semantic_lowobs_mass", pd.Series(0, index=labelled.index))).fillna(0.0) + 1.0), f"{signal}:confidence_shuffle"),
        "semantic_type_shuffle_within_pair": _shuffle(v, f"{signal}:semantic_type_shuffle"),
        "same_count_random_mode": pd.Series(0.0, index=labelled.index),
        "mode_mass_shuffle": _shuffle(v, f"{signal}:mode_mass_shuffle"),
        "sequence_label_shuffle": _shuffle(v, f"{signal}:sequence_label_shuffle"),
    }
    order = sorted(list(labelled.index), key=lambda i: stable_hash_float(signal, i))
    controls["same_count_random_mode"].loc[order[: int(flags.sum())]] = 1.0
    control_rhos = {name: spearman_rho(ctrl.tolist(), y.tolist()) for name, ctrl in controls.items()}
    max_control = max([x for x in control_rhos.values() if x is not None], default=None)
    margin = None if rho is None or max_control is None else float(rho - max_control)
    bad_or_high = bad | high
    bad_recall = float((flags & bad_or_high).sum() / max(int(bad_or_high.sum()), 1))
    good_fpr = float((flags & good_low).sum() / max(int(good_low.sum()), 1))
    good_any_fpr = float((flags & good).sum() / max(int(good.sum()), 1))
    seq01_stress = float((flags & labelled["seq"].astype(str).eq("01")).sum() / max(int(flags.sum()), 1))
    semantic_signal = signal.startswith("S") or signal.startswith("C")
    criterion_a = bool(semantic_signal and rho is not None and geometry_ref.get("rho") is not None and rho >= geometry_ref["rho"] + 0.05 and rho >= 0.30)
    criterion_b = bool(semantic_signal and margin is not None and margin >= 0.05 and bad_recall >= 0.60 and good_fpr <= 0.25)
    criterion_c = bool(
        semantic_signal
        and bad_recall >= (geometry_ref.get("bad_recall") or 0.0) - 0.05
        and good_fpr <= (geometry_ref.get("good_fpr") or 1.0) - 0.15
        and good_fpr <= 0.25
    )
    criterion_d = bool(semantic_signal and signal in {"S0_semantic_valid_dominant_mode", "S3_semantic_entropy_reduction"} and rho is not None and geometry_ref.get("rho") is not None and rho >= geometry_ref["rho"] + 0.05 and float(_num(labelled["semantic_entropy_reduction"]).fillna(0.0).mean()) > 0)
    signal_pass = bool(criterion_a or criterion_b or criterion_c or criterion_d)
    return {
        "signal": signal,
        "is_semantic_conditioned": semantic_signal,
        "available_rows": int(len(labelled)),
        "sequence_coverage": int(labelled["seq"].astype(str).str.zfill(2).nunique()),
        "spearman_rho_abs_log_scale_jump": rho,
        "max_control_rho": max_control,
        "semantic_shuffle_margin": margin,
        "bad_recall": bad_recall,
        "good_false_positive_rate": good_fpr,
        "good_any_fpr": good_any_fpr,
        "balanced_accuracy": float(0.5 * (bad_recall + (1.0 - good_fpr))),
        "signal_threshold_q75": threshold,
        "scale_high_threshold_q75": scale_threshold,
        "seq01_stress_flagged_fraction": seq01_stress,
        "criterion_A_semantic_lift": criterion_a,
        "criterion_B_semantic_specificity": criterion_b,
        "criterion_C_good_protection": criterion_c,
        "criterion_D_mode_disambiguation": criterion_d,
        "signal_pass": signal_pass,
    }


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    df = _load_rows(args.ledger_dir, args.v88_relevance_rows)
    df, filters = _apply_filter(df, args.filter)
    score_map = _scores(df)
    geometry_metrics: list[dict[str, Any]] = []
    dummy = {"rho": -1e9, "bad_recall": 0.0, "good_fpr": 1.0}
    for name in ["G0_geometry_dominant_mode", "G1_v88_overlap_baseline", "G2_mode_entropy_mad"]:
        geometry_metrics.append(_metric(df, name, score_map[name], dummy))
    geometry_ref = sorted(
        geometry_metrics,
        key=lambda row: (row["spearman_rho_abs_log_scale_jump"] if row["spearman_rho_abs_log_scale_jump"] is not None else -1e9, row["bad_recall"], -row["good_false_positive_rate"]),
        reverse=True,
    )[0]
    geometry_ref_simple = {"rho": geometry_ref["spearman_rho_abs_log_scale_jump"], "bad_recall": geometry_ref["bad_recall"], "good_fpr": geometry_ref["good_false_positive_rate"]}
    metric_rows = geometry_metrics + [_metric(df, name, vals, geometry_ref_simple) for name, vals in score_map.items() if not name.startswith("G")]
    for row in metric_rows:
        row["global_gate_components_pass"] = bool(
            row["sequence_coverage"] >= 3
            and row["available_rows"] >= 12
            and row["seq01_stress_flagged_fraction"] <= 0.50
            and (not row["is_semantic_conditioned"] or (row["semantic_shuffle_margin"] is not None and row["semantic_shuffle_margin"] >= 0.05))
        )
        row["phase2_gate_signal_pass"] = bool(row["signal_pass"] and row["global_gate_components_pass"])
    pass_rows = [row for row in metric_rows if row["is_semantic_conditioned"] and row["phase2_gate_signal_pass"]]
    best_sem = sorted(
        [row for row in metric_rows if row["is_semantic_conditioned"]],
        key=lambda row: (bool(row["phase2_gate_signal_pass"]), row["spearman_rho_abs_log_scale_jump"] if row["spearman_rho_abs_log_scale_jump"] is not None else -1e9, row["bad_recall"], -row["good_false_positive_rate"]),
        reverse=True,
    )[0]
    rows_out = df.copy()
    for name, vals in score_map.items():
        rows_out[name] = vals
    summary = {
        "phase": "Phase2_semantic_mode_relevance",
        "filter": args.filter,
        "filters": filters,
        "phase2_semantic_mode_relevance_gate_pass": len(pass_rows) > 0,
        "passing_semantic_signals": [row["signal"] for row in pass_rows],
        "geometry_reference_signal": geometry_ref["signal"],
        "geometry_reference_rho": geometry_ref["spearman_rho_abs_log_scale_jump"],
        "geometry_reference_bad_recall": geometry_ref["bad_recall"],
        "geometry_reference_good_fpr": geometry_ref["good_false_positive_rate"],
        "best_semantic_signal": best_sem,
        "scale_label_rows": int(_num(df["abs_log_scale_jump_gt"]).notna().sum()),
        "sequence_coverage": int(df[_num(df["abs_log_scale_jump_gt"]).notna()]["seq"].astype(str).str.zfill(2).nunique()),
        "semantic_valid_support_pair_mass_nonzero_rows": int((_num(df["semantic_valid_mass"]).fillna(0.0) > 0).sum()),
        "runtime_action_allowed": False,
        "ttt_allowed": False,
        "note": "Offline scale labels are audit-only. Semantic scores are fixed deterministic functions of Phase1 semantic mode ledger.",
    }
    if not pass_rows:
        summary["blocker"] = "no_semantic_conditioned_mode_signal_passed"
    write_csv(args.out_dir / "semantic_mode_relevance_by_signal.csv", metric_rows)
    write_csv(args.out_dir / "semantic_mode_relevance_rows.csv", rows_out.to_dict("records"))
    control_rows = [
        {"control_family": "label_shuffle", "status": "computed_per_signal"},
        {"control_family": "confidence_shuffle", "status": "computed_per_signal"},
        {"control_family": "semantic_type_shuffle_within_pair", "status": "computed_per_signal"},
        {"control_family": "same_count_random_mode", "status": "computed_per_signal"},
        {"control_family": "mode_mass_shuffle", "status": "computed_per_signal"},
        {"control_family": "sequence_label_shuffle", "status": "computed_per_signal"},
    ]
    write_csv(args.out_dir / "semantic_mode_relevance_controls.csv", control_rows)
    write_json(args.out_dir / "semantic_mode_relevance_summary.json", summary)
    report = [
        "# v89 Phase2 Semantic Mode Relevance",
        "",
        f"- filter: `{args.filter}`",
        f"- phase2_semantic_mode_relevance_gate_pass: `{summary['phase2_semantic_mode_relevance_gate_pass']}`",
        f"- passing_semantic_signals: `{summary['passing_semantic_signals']}`",
        f"- geometry_reference_signal: `{summary['geometry_reference_signal']}` rho=`{summary['geometry_reference_rho']}`",
        f"- best_semantic_signal: `{best_sem['signal']}` rho=`{best_sem['spearman_rho_abs_log_scale_jump']}` margin=`{best_sem['semantic_shuffle_margin']}` recall=`{best_sem['bad_recall']}` good_fpr=`{best_sem['good_false_positive_rate']}`",
        f"- semantic_valid_support_pair_mass_nonzero_rows: `{summary['semantic_valid_support_pair_mass_nonzero_rows']}`",
        f"- blocker: `{summary.get('blocker', '')}`",
    ]
    (args.out_dir / "semantic_mode_relevance_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"phase2_semantic_mode_relevance_gate_pass={summary['phase2_semantic_mode_relevance_gate_pass']}")
    print(f"passing_semantic_signals={summary['passing_semantic_signals']}")
    print(f"geometry_reference_signal={summary['geometry_reference_signal']}")
    print(f"geometry_reference_rho={summary['geometry_reference_rho']}")
    print(f"best_semantic_signal={best_sem['signal']}")
    print(f"best_semantic_rho={best_sem['spearman_rho_abs_log_scale_jump']}")
    print(f"best_semantic_margin={best_sem['semantic_shuffle_margin']}")
    print(f"best_semantic_recall={best_sem['bad_recall']}")
    print(f"best_semantic_good_fpr={best_sem['good_false_positive_rate']}")
    print(f"scale_label_rows={summary['scale_label_rows']}")
    print(f"sequence_coverage={summary['sequence_coverage']}")
    print(f"semantic_valid_support_pair_mass_nonzero_rows={summary['semantic_valid_support_pair_mass_nonzero_rows']}")
    if summary.get("blocker"):
        print(f"blocker={summary['blocker']}")


if __name__ == "__main__":
    main()
