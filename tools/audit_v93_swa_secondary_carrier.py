#!/usr/bin/env python3
"""Audit v93 Phase7 SWA secondary query/pair carrier smokes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.v86_soft_latent_utils import safe_float, write_csv, write_json  # noqa: E402
from tools.v93_semantic_object_identity_utils import ROOT  # noqa: E402


PHASE7_ROOT = ROOT / "phase7_swa_secondary_carrier"
ROUTE_ROOT = PHASE7_ROOT / "route_audit"
MASK_SUMMARY = PHASE7_ROOT / "route_masks/materialization_summary.json"
PHASE2_AUDIT = ROOT / "phase2_object_topology_policy/object_topology_policy_audit.json"
RUN_CASES = {
    "query": "P9_52_ATTENTION_BIAS_V92_POLICY_QUERY_MASS_AUDIT_LAST",
    "pair": "P9_54_ATTENTION_BIAS_V92_POLICY_PAIR_MASS_AUDIT_LAST",
}
SEQ_SPECS = {
    "00": {"chunk": 2},
    "01": {"chunk": 9},
    "02": {"chunk": 7},
    "05": {"chunk": 2},
}
VARIANTS = ["actual", "object", "component", "semantic", "regime", "random", "geometry"]
CONTROLS = ["object", "component", "semantic", "regime", "random", "geometry"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--route-root", type=Path, default=ROUTE_ROOT)
    parser.add_argument("--out-dir", type=Path, default=PHASE7_ROOT / "carrier_audit")
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        return {"read_error": str(exc), "path": str(path)}
    return obj if isinstance(obj, dict) else {"value": obj}


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


def _max(values: list[Any]) -> float | None:
    vals = [safe_float(v) for v in values]
    clean = [float(v) for v in vals if v is not None]
    if not clean:
        return None
    return float(max(clean))


def _metric_row(metrics_path: Path, run: str) -> dict[str, Any]:
    if not metrics_path.exists():
        return {"run": run, "missing_metrics": True}
    try:
        df = pd.read_csv(metrics_path)
    except Exception as exc:  # noqa: BLE001
        return {"run": run, "missing_metrics": True, "read_error": type(exc).__name__}
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


def _route_summary(route: str, df: pd.DataFrame) -> dict[str, Any]:
    actual = df[(df["route"].eq(route)) & (df["control"].eq("actual"))]
    actual_valid = actual[~actual["missing_metrics"]]
    random_df = df[(df["route"].eq(route)) & (df["control"].eq("random"))]
    actual_lifts = [float(v) for v in actual["selected_lift"].dropna().tolist()]
    actual_head = [float(v) for v in actual["headmax_lift"].dropna().tolist()]
    random_lifts = [float(v) for v in random_df["selected_lift"].dropna().tolist()]
    random_p95 = _p95(random_lifts)
    actual_mean = _mean(actual_lifts)
    actual_fidelity_count = int(actual_valid["action_fidelity_proxy"].sum()) if not actual_valid.empty else 0
    summary: dict[str, Any] = {
        "route": route,
        "seq_coverage": int(actual_valid["seq"].nunique()),
        "missing_metric_rows": int(df[df["route"].eq(route)]["missing_metrics"].sum()),
        "action_fidelity_actual": bool(not actual_valid.empty and actual_valid["action_fidelity_proxy"].all()),
        "action_fidelity_actual_count": actual_fidelity_count,
        "action_fidelity_actual_fraction": float(actual_fidelity_count / max(1, len(actual_valid))),
        "actual_route_lift_mean": actual_mean,
        "actual_route_lift_max": _max(actual_lifts),
        "actual_headmax_lift_mean": _mean(actual_head),
        "random_route_lift_mean": _mean(random_lifts),
        "random_route_lift_p95": random_p95,
        "actual_minus_random_p95": actual_mean - random_p95 if actual_mean is not None and random_p95 is not None else None,
        "selected_after_max": _max(df[df["route"].eq(route)]["selected_after"].dropna().tolist()),
        "source_after_max": _max(df[df["route"].eq(route)]["source_after"].dropna().tolist()),
        "selected_after_mean": _mean(df[df["route"].eq(route)]["selected_after"].dropna().tolist()),
        "source_after_mean": _mean(df[df["route"].eq(route)]["source_after"].dropna().tolist()),
        "by_seq": [],
    }
    for control in CONTROLS:
        cdf = df[(df["route"].eq(route)) & (df["control"].eq(control))]
        control_mean = _mean(cdf["selected_lift"].dropna().tolist())
        summary[f"{control}_route_lift_mean"] = control_mean
        summary[f"{control}_margin"] = actual_mean - control_mean if actual_mean is not None and control_mean is not None else None
    for seq in sorted(set(actual["seq"].astype(str))):
        a_rows = actual[actual["seq"].astype(str).eq(seq)]
        item: dict[str, Any] = {
            "seq": seq,
            "actual_selected_lift": safe_float(a_rows["selected_lift"].iloc[0]) if not a_rows.empty else None,
        }
        for control in CONTROLS:
            c_rows = df[(df["route"].eq(route)) & (df["control"].eq(control)) & (df["seq"].astype(str).eq(seq))]
            c_value = safe_float(c_rows["selected_lift"].iloc[0]) if not c_rows.empty else None
            item[f"{control}_selected_lift"] = c_value
            item[f"actual_minus_{control}"] = (
                item["actual_selected_lift"] - c_value
                if item["actual_selected_lift"] is not None and c_value is not None
                else None
            )
        summary["by_seq"].append(item)
    return summary


def _gate(summary: dict[str, Any], *, bad_recall: float | None, good_fpr: float | None, collapse_flag: bool) -> tuple[bool, dict[str, bool]]:
    checks = {
        "actual_minus_random_p95_ge_0_05": (summary.get("actual_minus_random_p95") or 0.0) >= 0.05,
        "object_margin_ge_0_05": (summary.get("object_margin") or 0.0) >= 0.05,
        "component_margin_ge_0_05": (summary.get("component_margin") or 0.0) >= 0.05,
        "semantic_margin_ge_0_05": (summary.get("semantic_margin") or 0.0) >= 0.05,
        "regime_margin_ge_0_05": (summary.get("regime_margin") or 0.0) >= 0.05,
        "geometry_margin_ge_0_05": (summary.get("geometry_margin") or 0.0) >= 0.05,
        "row_entropy_drop_available": False,
        "row_entropy_drop_le_0_20": False,
        "collapse_false": not collapse_flag,
        "bad_route_recall_ge_0_50": (bad_recall or 0.0) >= 0.50,
        "good_route_fpr_le_0_25": (good_fpr if good_fpr is not None else 1.0) <= 0.25,
        "coverage_ge_3": int(summary.get("seq_coverage") or 0) >= 3,
        "action_fidelity_actual": bool(summary.get("action_fidelity_actual")),
    }
    return bool(all(checks.values())), checks


def main() -> None:
    args = parse_args()
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for variant in VARIANTS:
        for route, run in RUN_CASES.items():
            for seq, spec in SEQ_SPECS.items():
                chunk = int(spec["chunk"])
                root = args.route_root / f"seq{seq}_chunk{chunk:02d}_v93_{variant}_{route}"
                rows.append(_row(seq, route, variant, root, run))
    write_csv(out / "phase7_swa_secondary_carrier_rows.csv", rows)
    df = pd.DataFrame(rows)

    phase2 = _read_json(PHASE2_AUDIT)
    actual_policy = phase2.get("actual_policy") if isinstance(phase2.get("actual_policy"), dict) else {}
    bad_recall = safe_float(actual_policy.get("bad_recall"))
    good_fpr = safe_float(actual_policy.get("good_FPR"))
    mask_summary = _read_json(MASK_SUMMARY)

    query_summary = _route_summary("query", df)
    pair_summary = _route_summary("pair", df)
    selected_after_max = _max(df["selected_after"].dropna().tolist())
    source_after_max = _max(df["source_after"].dropna().tolist())
    collapse_flag = bool(
        (selected_after_max is not None and selected_after_max >= 0.95)
        or (source_after_max is not None and source_after_max >= 0.95)
    )
    query_gate, query_checks = _gate(query_summary, bad_recall=bad_recall, good_fpr=good_fpr, collapse_flag=collapse_flag)
    pair_gate, pair_checks = _gate(pair_summary, bad_recall=bad_recall, good_fpr=good_fpr, collapse_flag=collapse_flag)
    gate_pass = bool(query_gate or pair_gate)
    summary = {
        "phase": "Phase7_v93_swa_secondary_carrier_audit",
        "entered": True,
        "entry_reason": "Phase5 counterfactual upper bound failed; plan line 1096-1097 directs SWA secondary/action-surface rediscovery when no counterfactual moves geometry.",
        "phase7_swa_secondary_carrier_gate_pass": gate_pass,
        "query_gate_pass": bool(query_gate),
        "pair_gate_pass": bool(pair_gate),
        "blocker": None if gate_pass else "phase7_swa_secondary_route_lift_tiny_or_not_control_specific_or_entropy_unmeasured",
        "route_dump_seq_coverage": int(max(query_summary.get("seq_coverage", 0), pair_summary.get("seq_coverage", 0))),
        "missing_metric_rows": int(df["missing_metrics"].sum()),
        "materialization_feasible": bool(mask_summary.get("materialization_feasible")),
        "bad_route_recall_from_phase2": bad_recall,
        "good_route_FPR_from_phase2": good_fpr,
        "row_entropy_drop_available": False,
        "row_entropy_drop_gate_pass": False,
        "collapse_flag": collapse_flag,
        "selected_after_max": selected_after_max,
        "source_after_max": source_after_max,
        "query": query_summary,
        "pair": pair_summary,
        "query_gate_checks": query_checks,
        "pair_gate_checks": pair_checks,
        "runtime_action_allowed": False,
        "ttt_allowed": False,
        "no_success_claim": "This is a SWA route/control audit only. Runtime action and TTT remain closed unless a separate runtime gate is passed.",
    }
    write_json(out / "phase7_swa_secondary_summary.json", summary)
    (out / "phase7_swa_secondary_report.md").write_text(
        "\n".join(
            [
                "# v93 Phase7 SWA Secondary Carrier Audit",
                "",
                f"phase7_swa_secondary_carrier_gate_pass: `{summary['phase7_swa_secondary_carrier_gate_pass']}`",
                f"blocker: `{summary['blocker']}`",
                f"missing_metric_rows: `{summary['missing_metric_rows']}`",
                f"collapse_flag: `{summary['collapse_flag']}`",
                f"row_entropy_drop_available: `{summary['row_entropy_drop_available']}`",
                "",
                "## Query Route",
                "",
                f"actual_minus_random_p95: `{query_summary.get('actual_minus_random_p95')}`",
                f"object_margin: `{query_summary.get('object_margin')}`",
                f"component_margin: `{query_summary.get('component_margin')}`",
                f"semantic_margin: `{query_summary.get('semantic_margin')}`",
                f"geometry_margin: `{query_summary.get('geometry_margin')}`",
                "",
                "## Pair Route",
                "",
                f"actual_minus_random_p95: `{pair_summary.get('actual_minus_random_p95')}`",
                f"object_margin: `{pair_summary.get('object_margin')}`",
                f"component_margin: `{pair_summary.get('component_margin')}`",
                f"semantic_margin: `{pair_summary.get('semantic_margin')}`",
                f"geometry_margin: `{pair_summary.get('geometry_margin')}`",
                "",
                "No runtime action or TTT write is allowed from this route audit.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"phase7_swa_secondary_carrier_gate_pass={gate_pass}")
    print(f"query_actual_minus_random_p95={query_summary.get('actual_minus_random_p95')}")
    print(f"pair_actual_minus_random_p95={pair_summary.get('actual_minus_random_p95')}")
    print(f"query_object_margin={query_summary.get('object_margin')}")
    print(f"pair_object_margin={pair_summary.get('object_margin')}")
    print(f"missing_metric_rows={summary['missing_metric_rows']}")
    print("runtime_action_allowed=False")
    print("ttt_allowed=False")


if __name__ == "__main__":
    main()
