#!/usr/bin/env python3
"""Audit v92 Phase4 SWA semantic-policy query/pair carrier smokes."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from v86_soft_latent_utils import safe_float, write_csv, write_json
from v92_semantic_policy_carrier_utils import ROOT


PHASE4_ROOT = ROOT / "phase4_swa_semantic_policy_route_audit"
PHASE1_AUDIT = ROOT / "phase1_semantic_policy_row_bank/phase1_gate_summary.json"
SEQ_ROOTS = {
    "00": PHASE4_ROOT / "seq00_chunk02_v92_policy_query_pair",
    "01": PHASE4_ROOT / "seq01_chunk09_v92_policy_query_pair",
    "02": PHASE4_ROOT / "seq02_chunk07_v92_policy_query_pair",
    "05": PHASE4_ROOT / "seq05_chunk02_v92_policy_query_pair",
}
QUERY_ACTUAL = "P9_52_ATTENTION_BIAS_V92_POLICY_QUERY_MASS_AUDIT_LAST"
QUERY_RANDOM = "P9_53_ATTENTION_BIAS_V92_POLICY_QUERY_RANDOM_SAME_MASS_MASS_AUDIT_LAST"
PAIR_ACTUAL = "P9_54_ATTENTION_BIAS_V92_POLICY_PAIR_MASS_AUDIT_LAST"
PAIR_RANDOM = "P9_55_ATTENTION_BIAS_V92_POLICY_PAIR_RANDOM_SAME_MASS_MASS_AUDIT_LAST"
PAIR_CONTROLS = {
    "semantic_shuffle": PHASE4_ROOT / "semantic_shuffle_controls",
    "component_shuffle": PHASE4_ROOT / "component_shuffle_controls",
    "regime_shuffle": PHASE4_ROOT / "regime_shuffle_controls",
    "geometry_only": PHASE4_ROOT / "geometry_shuffle_controls",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase4-root", type=Path, default=PHASE4_ROOT)
    parser.add_argument("--out-dir", type=Path, default=PHASE4_ROOT / "phase4_swa_carrier_audit")
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    import json

    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        obj = json.load(handle)
    return obj if isinstance(obj, dict) else {}


def _p95(values: list[float]) -> float | None:
    vals = sorted(float(v) for v in values if v is not None)
    if not vals:
        return None
    if len(vals) == 1:
        return vals[0]
    pos = 0.95 * (len(vals) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(vals) - 1)
    frac = pos - lo
    return float(vals[lo] * (1.0 - frac) + vals[hi] * frac)


def _mean(values: list[Any]) -> float | None:
    vals = [safe_float(v) for v in values]
    clean = [float(v) for v in vals if v is not None]
    if not clean:
        return None
    return float(sum(clean) / len(clean))


def _metric_row(metrics_path: Path, run: str) -> dict[str, Any]:
    if not metrics_path.exists():
        return {"run": run, "missing_metrics": True}
    df = pd.read_csv(metrics_path)
    rows = df[df["run"].astype(str).eq(run)]
    if rows.empty:
        return {"run": run, "missing_run": True}
    row = rows.iloc[0].to_dict()
    row["missing_metrics"] = False
    return row


def _row(seq: str, route: str, control: str, root: Path, run: str) -> dict[str, Any]:
    metrics_path = root / "phase9_swa_cache_value_metrics.csv"
    row = _metric_row(metrics_path, run)
    return {
        "seq": seq,
        "route": route,
        "control": control,
        "run": run,
        "metrics_path": str(metrics_path),
        "run_dir": row.get("run_dir", ""),
        "missing_metrics": bool(row.get("missing_metrics") or row.get("missing_run")),
        "action_fidelity_proxy": bool(
            (safe_float(row.get("phase9_swa_overlap_bias_applied_sum")) or 0.0) > 0.0
            and (safe_float(row.get("phase9_swa_overlap_bias_mean_abs")) or 0.0) > 0.0
        ),
        "attention_mass_available_frac": safe_float(row.get("phase9_swa_attention_mass_available_frac")),
        "selected_lift": safe_float(row.get("phase9_swa_attention_mass_selected_lift")),
        "source_lift": safe_float(row.get("phase9_swa_attention_mass_source_lift")),
        "headmax_lift": safe_float(row.get("phase9_swa_attention_mass_selected_head_max_lift")),
        "selected_before": safe_float(row.get("phase9_swa_attention_mass_selected_before")),
        "selected_after": safe_float(row.get("phase9_swa_attention_mass_selected_after")),
        "source_before": safe_float(row.get("phase9_swa_attention_mass_source_before")),
        "source_after": safe_float(row.get("phase9_swa_attention_mass_source_after")),
        "mean_abs_bias": safe_float(row.get("phase9_swa_overlap_bias_mean_abs")),
        "max_abs_bias": safe_float(row.get("phase9_swa_overlap_bias_max_abs")),
        "local_sim3_ate_rmse_m": safe_float(row.get("local_sim3_ate_rmse_m")),
        "overlap3_to_future_pose_sim3_rmse_m": safe_float(row.get("overlap3_to_future_pose_sim3_rmse_m")),
        "scale_cv_head_mid_tail_pose_sim3": safe_float(row.get("scale_cv_head_mid_tail_pose_sim3")),
    }


def _seq_from_control_root(path: Path) -> str:
    name = path.name
    if name.startswith("seq") and "_chunk" in name:
        return name[3:5]
    return ""


def main() -> None:
    args = parse_args()
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for seq, root in SEQ_ROOTS.items():
        rows.append(_row(seq, "query", "actual", root, QUERY_ACTUAL))
        rows.append(_row(seq, "query", "random_same_mass", root, QUERY_RANDOM))
        rows.append(_row(seq, "pair", "actual", root, PAIR_ACTUAL))
        rows.append(_row(seq, "pair", "random_same_mass", root, PAIR_RANDOM))

    for control_name, root in PAIR_CONTROLS.items():
        for child in sorted(root.glob("seq*_chunk*_v92_*_pair")):
            seq = _seq_from_control_root(child)
            rows.append(_row(seq, "pair", control_name, child, PAIR_ACTUAL))

    write_csv(out / "phase4_swa_carrier_rows.csv", rows)

    df = pd.DataFrame(rows)
    actual_random_success = int((~df[df["control"].isin(["actual", "random_same_mass"])]["missing_metrics"]).sum())
    control_success = int((~df[df["control"].isin(PAIR_CONTROLS.keys())]["missing_metrics"]).sum())
    missing_count = int(df["missing_metrics"].sum())

    phase1 = _read_json(PHASE1_AUDIT)
    bad_recall = safe_float(phase1.get("bad_recall"))
    good_fpr = safe_float(phase1.get("good_FPR"))

    query_actual = df[(df["route"] == "query") & (df["control"] == "actual")]
    query_random = df[(df["route"] == "query") & (df["control"] == "random_same_mass")]
    pair_actual = df[(df["route"] == "pair") & (df["control"] == "actual")]
    pair_random = df[(df["route"] == "pair") & (df["control"] == "random_same_mass")]

    def _route_summary(route: str, actual: pd.DataFrame, random_df: pd.DataFrame) -> dict[str, Any]:
        actual_lifts = [float(v) for v in actual["selected_lift"].dropna().tolist()]
        random_lifts = [float(v) for v in random_df["selected_lift"].dropna().tolist()]
        actual_head = [float(v) for v in actual["headmax_lift"].dropna().tolist()]
        random_head = [float(v) for v in random_df["headmax_lift"].dropna().tolist()]
        by_seq = []
        for seq in sorted(set(actual["seq"].astype(str)) & set(random_df["seq"].astype(str))):
            a = safe_float(actual[actual["seq"].astype(str).eq(seq)]["selected_lift"].iloc[0])
            r = safe_float(random_df[random_df["seq"].astype(str).eq(seq)]["selected_lift"].iloc[0])
            ah = safe_float(actual[actual["seq"].astype(str).eq(seq)]["headmax_lift"].iloc[0])
            rh = safe_float(random_df[random_df["seq"].astype(str).eq(seq)]["headmax_lift"].iloc[0])
            by_seq.append(
                {
                    "seq": seq,
                    "selected_lift_actual_minus_random": (a - r) if a is not None and r is not None else None,
                    "headmax_lift_actual_minus_random": (ah - rh) if ah is not None and rh is not None else None,
                }
            )
        random_p95 = _p95(random_lifts)
        return {
            "route": route,
            "seq_coverage": int(actual["seq"].nunique()),
            "action_fidelity_all": bool(actual["action_fidelity_proxy"].all() and random_df["action_fidelity_proxy"].all()),
            "actual_route_lift_mean": _mean(actual_lifts),
            "random_route_lift_mean": _mean(random_lifts),
            "random_route_lift_p95": random_p95,
            "actual_minus_random_p95": (
                _mean(actual_lifts) - random_p95 if _mean(actual_lifts) is not None and random_p95 is not None else None
            ),
            "actual_beats_random_count": int(sum(1 for item in by_seq if (item["selected_lift_actual_minus_random"] or 0.0) > 0.0)),
            "actual_headmax_lift_mean": _mean(actual_head),
            "random_headmax_lift_mean": _mean(random_head),
            "actual_headmax_beats_random_count": int(sum(1 for item in by_seq if (item["headmax_lift_actual_minus_random"] or 0.0) > 0.0)),
            "by_seq": by_seq,
        }

    query_summary = _route_summary("query", query_actual, query_random)
    pair_summary = _route_summary("pair", pair_actual, pair_random)

    pair_actual_mean = pair_summary["actual_route_lift_mean"]
    for control_name in ["semantic_shuffle", "component_shuffle", "regime_shuffle", "geometry_only"]:
        control_df = df[(df["route"] == "pair") & (df["control"] == control_name)]
        control_mean = _mean(control_df["selected_lift"].dropna().tolist())
        pair_summary[f"{control_name}_route_lift_mean"] = control_mean
        pair_summary[f"{control_name}_margin"] = (
            pair_actual_mean - control_mean if pair_actual_mean is not None and control_mean is not None else None
        )

    max_selected_after = _mean(df[df["route"].eq("pair")]["selected_after"].dropna().tolist())
    max_source_after = _mean(df[df["route"].eq("pair")]["source_after"].dropna().tolist())
    collapse_flag = bool(
        (max_selected_after is not None and max_selected_after >= 0.95)
        or (max_source_after is not None and max_source_after >= 0.95)
    )

    query_gate = bool(
        (query_summary.get("actual_minus_random_p95") or 0.0) >= 0.05
        and (bad_recall or 0.0) >= 0.50
        and (good_fpr if good_fpr is not None else 1.0) <= 0.25
        and query_summary.get("seq_coverage", 0) >= 3
        and not collapse_flag
    )
    pair_gate = bool(
        (pair_summary.get("actual_minus_random_p95") or 0.0) >= 0.05
        and (pair_summary.get("semantic_shuffle_margin") or 0.0) >= 0.05
        and (pair_summary.get("component_shuffle_margin") or 0.0) >= 0.05
        and (pair_summary.get("regime_shuffle_margin") or 0.0) >= 0.05
        and (bad_recall or 0.0) >= 0.50
        and (good_fpr if good_fpr is not None else 1.0) <= 0.25
        and pair_summary.get("seq_coverage", 0) >= 3
        and not collapse_flag
    )

    summary = {
        "phase": "Phase4_swa_semantic_policy_carrier_audit",
        "phase4_swa_carrier_gate_pass": bool(query_gate or pair_gate),
        "query_gate_pass": bool(query_gate),
        "pair_gate_pass": bool(pair_gate),
        "blocker": None
        if query_gate or pair_gate
        else "phase4_swa_query_pair_route_lift_tiny_or_not_shuffle_specific",
        "actual_random_successful_rows": actual_random_success,
        "pair_control_successful_rows": control_success,
        "missing_metric_rows": missing_count,
        "route_dump_seq_coverage": int(max(query_summary.get("seq_coverage", 0), pair_summary.get("seq_coverage", 0))),
        "bad_recall_from_phase1": bad_recall,
        "good_FPR_from_phase1": good_fpr,
        "collapse_flag": collapse_flag,
        "row_entropy_drop_available": False,
        "query": query_summary,
        "pair": pair_summary,
        "runtime_action_allowed": False,
        "ttt_allowed": False,
    }
    write_json(out / "phase4_swa_carrier_summary.json", summary)
    (out / "phase4_swa_carrier_report.md").write_text(
        "\n".join(
            [
                "# v92 Phase4 SWA Carrier Audit",
                "",
                f"phase4_swa_carrier_gate_pass: `{summary['phase4_swa_carrier_gate_pass']}`",
                f"blocker: `{summary['blocker']}`",
                "",
                "## Query Route",
                "",
                f"actual_minus_random_p95: `{query_summary.get('actual_minus_random_p95')}`",
                f"actual_beats_random_count: `{query_summary.get('actual_beats_random_count')}/4`",
                "",
                "## Pair Route",
                "",
                f"actual_minus_random_p95: `{pair_summary.get('actual_minus_random_p95')}`",
                f"semantic_shuffle_margin: `{pair_summary.get('semantic_shuffle_margin')}`",
                f"component_shuffle_margin: `{pair_summary.get('component_shuffle_margin')}`",
                f"regime_shuffle_margin: `{pair_summary.get('regime_shuffle_margin')}`",
                f"geometry_only_margin: `{pair_summary.get('geometry_only_margin')}`",
                "",
                "No runtime action or TTT write is allowed from this Phase4 audit.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"phase4_swa_carrier_gate_pass={summary['phase4_swa_carrier_gate_pass']}")
    print(f"query_actual_minus_random_p95={query_summary.get('actual_minus_random_p95')}")
    print(f"pair_actual_minus_random_p95={pair_summary.get('actual_minus_random_p95')}")
    print(f"pair_semantic_shuffle_margin={pair_summary.get('semantic_shuffle_margin')}")
    print(f"pair_component_shuffle_margin={pair_summary.get('component_shuffle_margin')}")
    print(f"pair_regime_shuffle_margin={pair_summary.get('regime_shuffle_margin')}")


if __name__ == "__main__":
    main()
