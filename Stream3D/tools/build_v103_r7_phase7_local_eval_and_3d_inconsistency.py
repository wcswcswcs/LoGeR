#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


STREAM3D_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = STREAM3D_ROOT.parent
AUDIT_ROOT = STREAM3D_ROOT / "outputs" / "audit"
PHASE_ID = "v103_r7_phase7_local_eval_and_3d_inconsistency"
DEFAULT_OUT = AUDIT_ROOT / "v103_r7_phase7_local_eval_and_3d_inconsistency"


PHASE_INPUTS = [
    ("R7-3", "R7AS_anchor_confirmed_support", AUDIT_ROOT / "v103_r7_phase3_anchor_confirmed_support"),
    ("R7-4", "R7SS_skeleton_confirmed_support", AUDIT_ROOT / "v103_r7_phase4_skeleton_confirmed_support"),
    ("R7-5", "R7SP_anchor_seeded_support_propagation", AUDIT_ROOT / "v103_r7_phase5_anchor_seeded_support_propagation"),
]
MISSING_PHASES = [
    ("R7-2", "support_conditioned_features", "not_run_in_current_r7_ladder"),
    ("R7-6", "support_ranking_extent", "not_run_in_current_r7_ladder"),
]


def _rel(path: Path | str) -> str:
    p = Path(path)
    try:
        return p.resolve().relative_to(REPO_ROOT).as_posix()
    except Exception:
        return p.as_posix()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return _rel(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields or ["schema_version"])
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _jsonable(row.get(key, "")) for key in fields})


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _num(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        out = float(value)
        return out if np.isfinite(out) else default
    except Exception:
        return default


def _window_stats(window_rows: pd.DataFrame, variant_id: str) -> dict[str, Any]:
    if window_rows.empty or "variant_id" not in window_rows:
        return {}
    rows = window_rows[window_rows["variant_id"].astype(str) == str(variant_id)]
    if rows.empty:
        return {}
    frag = rows.get("gt_fragment_count_mean", pd.Series(dtype=float)).astype(float).to_numpy()
    ge2 = rows.get("gt_fragment_count_ge2_rate", pd.Series(dtype=float)).astype(float).to_numpy()
    return {
        "GT_fragment_count_mean": float(np.mean(frag)) if frag.size else "",
        "GT_fragment_count_p50": float(np.quantile(frag, 0.50)) if frag.size else "",
        "GT_fragment_count_p90": float(np.quantile(frag, 0.90)) if frag.size else "",
        "GT_fragment_count_ge2_rate": float(np.mean(ge2)) if ge2.size else "",
    }


def _pick(row: pd.Series, *names: str, default: Any = "") -> Any:
    for name in names:
        if name in row and not pd.isna(row[name]):
            return row[name]
    return default


def build(args: argparse.Namespace) -> dict[str, Any]:
    t0 = time.time()
    out = Path(args.output_root)
    if not out.is_absolute():
        out = STREAM3D_ROOT / out
    if out.exists() and any(out.iterdir()) and not args.force:
        raise SystemExit(f"output root already exists and is non-empty; pass --force: {_rel(out)}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "last_command.txt").write_text(" ".join([sys.executable, str(Path(__file__)), *sys.argv[1:]]) + "\n", encoding="utf-8")

    artifact_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    variant_rows: list[dict[str, Any]] = []
    control_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    summaries: dict[str, dict[str, Any]] = {}

    for phase, family, root in PHASE_INPUTS:
        summary = _read_json(root / "summary.json")
        summaries[phase] = summary
        metric_path = root / "metric_rows.csv"
        window_path = root / "window_rows.csv"
        artifact_rows.append(
            {
                "schema_version": "stream4d_v103_r7_phase7_artifact_row_v1",
                "phase_id": PHASE_ID,
                "source_phase": phase,
                "artifact_role": "source_metric_rows",
                "path": _rel(metric_path),
                "exists": metric_path.exists(),
                "required": True,
                "uses_gt_for_prediction": False,
                "uses_future": False,
            }
        )
        if not metric_path.exists():
            failure_rows.append(
                {
                    "schema_version": "stream4d_v103_r7_phase7_failure_row_v1",
                    "phase_id": PHASE_ID,
                    "failure_id": "R7_SOURCE_METRIC_ROWS_MISSING",
                    "source_phase": phase,
                    "severity": "blocking",
                    "detail": _rel(metric_path),
                    "repair_direction": f"Run {phase} before formal R7-7 aggregation.",
                    "uses_gt_for_prediction": False,
                    "uses_future": False,
                }
            )
            continue
        df = pd.read_csv(metric_path)
        wdf = pd.read_csv(window_path) if window_path.exists() else pd.DataFrame()
        for item in df.to_dict("records"):
            row = pd.Series(item)
            variant_id = str(row.get("variant_id", ""))
            wstats = _window_stats(wdf, variant_id)
            out_row = {
                "schema_version": "stream4d_v103_r7_phase7_metric_row_v1",
                "phase_id": PHASE_ID,
                "source_phase": phase,
                "variant_id": variant_id,
                "base_variant_id": str(row.get("base_variant_id", variant_id)),
                "variant_family": str(row.get("variant_family", family)),
                "control_role": str(row.get("control_role", "real")),
                "MV_AP_window": _pick(row, "MV_AP_window", default=""),
                "MV_AP50_window": _pick(row, "MV_AP50_window", default=""),
                "MV_AP25_window": _pick(row, "MV_AP25_window", default=""),
                "ScoreFreeMatch50_window": _pick(row, "ScoreFreeMatch50_window", default=""),
                "ScoreFreeMatch25_window": _pick(row, "ScoreFreeMatch25_window", default=""),
                "same_frame_collision_count": _pick(row, "same_frame_collision_count", default=""),
                "pixel_collision_rate": _pick(row, "pixel_collision_rate", default=""),
                "missing_mask_raster_count": _pick(row, "missing_mask_raster_count", default=""),
                "uses_gt_for_prediction": bool(row.get("uses_gt_for_prediction", False)),
                "uses_gt_for_eval": bool(row.get("uses_gt_for_eval", True)),
                "uses_future": bool(row.get("uses_future", False)),
                "accepted_edge_count": _pick(row, "accepted_edge_count", default=""),
                "accepted_S_only_edge_count": _pick(row, "accepted_S_only_edge_count", default=""),
                "accepted_diff_gt_edge_count_diagnostic": _pick(row, "accepted_diff_gt_edge_count_diagnostic", default=""),
                "same_GT_connection_rate_diagnostic": _pick(row, "same_GT_connection_rate_diagnostic", default=""),
                "diff_GT_false_connection_rate_diagnostic": _pick(row, "diff_GT_false_connection_rate_diagnostic", default=""),
                "same_semantic_diff_GT_false_connection_rate_diagnostic": _pick(row, "same_semantic_diff_GT_false_connection_rate", default=""),
                "GT_fragment_count_mean": _pick(row, "GT_fragment_count_mean", default=wstats.get("GT_fragment_count_mean", "")),
                "GT_fragment_count_p50": wstats.get("GT_fragment_count_p50", ""),
                "GT_fragment_count_p90": wstats.get("GT_fragment_count_p90", ""),
                "GT_fragment_count_ge2_rate": _pick(row, "GT_fragment_count_ge2_rate", default=wstats.get("GT_fragment_count_ge2_rate", "")),
                "best_pred_IoU_mean": _pick(row, "best_pred_IoU", default=""),
                "union_pred_IoU_mean": _pick(row, "union_pred_IoU", default=""),
                "union_minus_best_IoU_mean": _pick(row, "union_minus_best_IoU", "union_minus_best_IoU_mean", default=""),
                "real_minus_best_control_MV_AP_window": _pick(row, "real_minus_best_control_MV_AP_window", default=""),
                "real_minus_best_control_MV_AP50_window": _pick(row, "real_minus_best_control_MV_AP50_window", default=""),
                "diagnostic_completeness": "edge_gt_diagnostic_available" if "same_GT_connection_rate_diagnostic" in row else "edge_gt_diagnostic_missing",
            }
            metric_rows.append(out_row)
            variant_rows.append(
                {
                    "schema_version": "stream4d_v103_r7_phase7_variant_row_v1",
                    "phase_id": PHASE_ID,
                    "source_phase": phase,
                    "variant_id": variant_id,
                    "variant_family": out_row["variant_family"],
                    "control_role": out_row["control_role"],
                    "uses_gt_for_prediction": False,
                    "uses_future": False,
                }
            )
            if out_row["control_role"] != "real":
                control_rows.append(out_row)

    for phase, family, reason in MISSING_PHASES:
        failure_rows.append(
            {
                "schema_version": "stream4d_v103_r7_phase7_failure_row_v1",
                "phase_id": PHASE_ID,
                "failure_id": "R7_SOURCE_PHASE_NOT_RUN",
                "source_phase": phase,
                "variant_family": family,
                "severity": "warning",
                "detail": reason,
                "repair_direction": "Do not fabricate rows; run the phase if R7-7 needs full R7-2..R7-6 coverage.",
                "uses_gt_for_prediction": False,
                "uses_future": False,
            }
        )

    real_rows = [r for r in metric_rows if str(r.get("control_role", "")) == "real"]
    best = max(real_rows, key=lambda r: (_num(r.get("MV_AP_window")), _num(r.get("MV_AP50_window"))), default={})
    no_promotion = not any(bool(s.get("phase_pass", False)) for s in summaries.values())
    gate_rows = [
        {
            "schema_version": "stream4d_v103_r7_phase7_gate_row_v1",
            "phase_id": PHASE_ID,
            "gate_name": "source_phases_r7_3_to_r7_5_available",
            "pass": all((root / "metric_rows.csv").exists() for _phase, _family, root in PHASE_INPUTS),
            "observed": [_rel(root / "metric_rows.csv") for _phase, _family, root in PHASE_INPUTS],
            "required": "R7-3/R7-4/R7-5 metric_rows",
            "repair_direction": "Run missing source phase.",
            "uses_gt_for_prediction": False,
            "uses_future": False,
        },
        {
            "schema_version": "stream4d_v103_r7_phase7_gate_row_v1",
            "phase_id": PHASE_ID,
            "gate_name": "no_subset_candidate_promoted",
            "pass": no_promotion,
            "observed": {phase: summary.get("phase_pass", "") for phase, summary in summaries.items()},
            "required": "true means R7-7 should not trigger R7-8 promotion automatically",
            "repair_direction": "Only proceed to R7-8 if a candidate has AP or 3D inconsistency evidence.",
            "uses_gt_for_prediction": False,
            "uses_future": False,
        },
        {
            "schema_version": "stream4d_v103_r7_phase7_gate_row_v1",
            "phase_id": PHASE_ID,
            "gate_name": "no_gt_prediction_leakage",
            "pass": not any(bool(r.get("uses_gt_for_prediction", False)) for r in metric_rows),
            "observed": "",
            "required": "all metric rows uses_gt_for_prediction=false",
            "repair_direction": "Remove contaminated rows.",
            "uses_gt_for_prediction": False,
            "uses_future": False,
        },
    ]
    if no_promotion:
        failure_rows.append(
            {
                "schema_version": "stream4d_v103_r7_phase7_failure_row_v1",
                "phase_id": PHASE_ID,
                "failure_id": "NO_R7_LOCAL_OR_HISTORY_CANDIDATE",
                "severity": "blocking",
                "detail": "R7-3/R7-4/R7-5 all failed subset promotion; R7-4 history_candidate=false and R7-5 produced no real propagation edge.",
                "repair_direction": "Either implement missing exact R7-1 intervention references/R7-2/R7-6, or stop before R7-8 history inheritance promotion.",
                "uses_gt_for_prediction": False,
                "uses_future": False,
            }
        )

    _write_csv(out / "artifact_rows.csv", artifact_rows)
    _write_csv(out / "variant_rows.csv", variant_rows)
    _write_csv(out / "metric_rows.csv", metric_rows)
    _write_csv(out / "control_rows.csv", control_rows)
    _write_csv(out / "gate_rows.csv", gate_rows)
    _write_csv(out / "failure_rows.csv", failure_rows)
    summary = {
        "schema_version": "stream4d_v103_r7_phase7_summary_v1",
        "phase": "R7-7",
        "phase_id": PHASE_ID,
        "phase_pass": False,
        "decision": "NO_GO_R7_7_NO_LOCAL_OR_HISTORY_CANDIDATE" if no_promotion else "PARTIAL_R7_7_REVIEW_CANDIDATE",
        "runtime_sec": time.time() - t0,
        "metric_row_count": len(metric_rows),
        "real_metric_row_count": len(real_rows),
        "control_row_count": len(control_rows),
        "best_real_variant_id": best.get("variant_id", ""),
        "best_real_variant_family": best.get("variant_family", ""),
        "best_real_MV_AP_window": best.get("MV_AP_window", ""),
        "best_real_MV_AP50_window": best.get("MV_AP50_window", ""),
        "r7_3_phase_pass": summaries.get("R7-3", {}).get("phase_pass", ""),
        "r7_4_phase_pass": summaries.get("R7-4", {}).get("phase_pass", ""),
        "r7_4_history_candidate": summaries.get("R7-4", {}).get("history_candidate", ""),
        "r7_5_phase_pass": summaries.get("R7-5", {}).get("phase_pass", ""),
        "uses_gt_for_prediction": False,
        "uses_gt_for_eval": True,
        "uses_future": False,
        "outputs": {
            "summary": _rel(out / "summary.json"),
            "gate_rows": _rel(out / "gate_rows.csv"),
            "failure_rows": _rel(out / "failure_rows.csv"),
            "variant_rows": _rel(out / "variant_rows.csv"),
            "metric_rows": _rel(out / "metric_rows.csv"),
            "control_rows": _rel(out / "control_rows.csv"),
            "artifact_rows": _rel(out / "artifact_rows.csv"),
        },
        "truthfulness_note": "R7-7 aggregates completed R7-3/R7-4/R7-5 outputs only. R7-2/R7-6 are marked missing rather than fabricated.",
    }
    _write_json(out / "summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate v103 R7 local eval and 3D inconsistency diagnostics.")
    parser.add_argument("--output-root", default=str(DEFAULT_OUT))
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    summary = build(parse_args())
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
