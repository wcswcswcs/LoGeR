#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


STREAM3D_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = STREAM3D_ROOT.parent
AUDIT_ROOT = STREAM3D_ROOT / "outputs" / "audit"

PHASE_ID = "v103_supp_r5_final_decision"
DEFAULT_OUT = AUDIT_ROOT / "v103_supp_r5_final_decision"
DEFAULT_FACT_LOCK_ROOT = AUDIT_ROOT / "v103_supp_r5_fact_lock"
DEFAULT_FEATURE_ROOT = AUDIT_ROOT / "v103_supp_r5_support_weighted_affinity"
DEFAULT_EDGE_ROOT = AUDIT_ROOT / "v103_supp_r5_support_edge_attribution"
DEFAULT_GT_ROOT = AUDIT_ROOT / "v103_supp_r5_gt_coverage"
DEFAULT_LOCAL_AP_ROOT = AUDIT_ROOT / "v103_supp_r5_support_weighted_local_ap_diag"
DEFAULT_ANCHOR_ONLY_ROOT = AUDIT_ROOT / "v103_supp_r5_anchor_only_local_ap_diag"


def _project(path: Path | str) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    if p.parts and p.parts[0] == "Stream3D":
        return REPO_ROOT / p
    return STREAM3D_ROOT / p


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
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _num(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        out = float(value)
        if not np.isfinite(out):
            return default
        return out
    except Exception:
        return default


def _claim_row(name: str, allowed: bool, evidence: str, boundary: str) -> dict[str, Any]:
    return {
        "schema_version": "stream4d_v103_supp_r5_final_decision_row_v1",
        "phase_id": PHASE_ID,
        "claim_name": name,
        "claim_allowed": bool(allowed),
        "evidence": evidence,
        "claim_boundary": boundary,
        "uses_gt_for_prediction": False,
        "uses_future": False,
    }


def _failure_rows(edge_root: Path, gt_root: Path, local_ap_root: Path, anchor_only_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source_name, root in [
        ("r5_2_edge_attribution", edge_root),
        ("r5_3_gt_diagnostic", gt_root),
        ("r5_4_support_weighted_local_ap", local_ap_root),
        ("r5_4_anchor_only_local_ap", anchor_only_root),
    ]:
        df = _read_csv(root / "failure_rows.csv")
        for _, rec in df.iterrows():
            rows.append(
                {
                    "schema_version": "stream4d_v103_supp_r5_final_failure_taxonomy_row_v1",
                    "phase_id": PHASE_ID,
                    "source": source_name,
                    "source_root": _rel(root),
                    "blocker": str(rec.get("blocker", rec.get("gate_name", ""))),
                    "detail": str(rec.get("detail", "")),
                    "repair_direction": str(rec.get("repair_direction", "")),
                    "uses_gt_for_prediction": False,
                    "uses_gt_for_eval": bool(source_name == "r5_3_gt_diagnostic" or "diagnostic" in source_name),
                    "uses_future": False,
                }
            )
    return rows


def _best_local_ap(local_ap_root: Path, anchor_only_root: Path) -> dict[str, Any]:
    dfs: list[pd.DataFrame] = []
    for root in [local_ap_root, anchor_only_root]:
        df = _read_csv(root / "variant_metric_rows.csv")
        if not df.empty:
            dfs.append(df)
    if not dfs:
        return {}
    df = pd.concat(dfs, ignore_index=True)
    real = df[~df["phase6d_variant_id"].astype(str).str.startswith("R")].copy()
    if real.empty:
        return {}
    idx = real["MV_AP_window"].astype(float).idxmax()
    return real.loc[idx].to_dict()


def _support_coverage(gt_root: Path) -> dict[str, Any]:
    df = _read_csv(gt_root / "gt_object_coverage_summary_rows.csv")
    if df.empty:
        return {}
    hit = df[df["group_key"].astype(str) == "all"]
    return hit.iloc[0].to_dict() if not hit.empty else {}


def _markdown(summary: dict[str, Any], decision_rows: list[dict[str, Any]], failures: list[dict[str, Any]]) -> str:
    lines = [
        "# Stream4D v103 R5 Method Claim Boundary",
        "",
        "This file is generated by `build_v103_supp_r5_final_decision.py`.",
        "",
        "## Decision",
        "",
        f"- overall_status: `{summary['overall_status']}`",
        f"- final_decision: `{summary['decision']}`",
        f"- local_subset_gate_pass: `{summary['local_subset_gate_pass']}`",
        f"- full_dev_allowed: `{summary['full_dev_allowed']}`",
        f"- history_allowed: `{summary['history_allowed']}`",
        "",
        "## Claim Boundary",
        "",
    ]
    for row in decision_rows:
        lines.append(f"- {row['claim_name']}: `{row['claim_allowed']}` - {row['claim_boundary']}")
    lines.extend(
        [
            "",
            "## Metric Boundary",
            "",
            "- Final local eval metric: `MV_AP_window` / `MV_AP50_window` from `Stream3D/tools/run_v65_scene_multiview_ap.py`.",
            "- Scene/local2history metric: `MV_AP_scene`; it was not run in R5 because local subset gate failed.",
            "- GT object coverage and 3D inconsistency are diagnostic-only and were not used to choose thresholds or variants.",
            "",
            "## Failure Taxonomy",
            "",
        ]
    )
    if failures:
        for row in failures:
            lines.append(f"- {row['source']}: `{row['blocker']}` {row['detail']}".rstrip())
    else:
        lines.append("- No failure rows were found, but final decision still depends on local/full-dev gates.")
    lines.append("")
    return "\n".join(lines)


def build(args: argparse.Namespace) -> dict[str, Any]:
    t0 = time.time()
    out = _project(args.output_root)
    if out.exists() and any(out.iterdir()) and not args.force:
        raise SystemExit(f"output root already exists and is non-empty; pass --force: {_rel(out)}")
    out.mkdir(parents=True, exist_ok=True)

    fact_root = _project(args.fact_lock_root)
    feature_root = _project(args.feature_root)
    edge_root = _project(args.edge_root)
    gt_root = _project(args.gt_root)
    local_ap_root = _project(args.local_ap_root)
    anchor_only_root = _project(args.anchor_only_root)

    fact_summary = _read_json(fact_root / "summary.json")
    feature_summary = _read_json(feature_root / "summary.json")
    edge_summary = _read_json(edge_root / "summary.json")
    gt_summary = _read_json(gt_root / "summary.json")
    local_summary = _read_json(local_ap_root / "summary.json")
    anchor_summary = _read_json(anchor_only_root / "summary.json")
    best = _best_local_ap(local_ap_root, anchor_only_root)
    coverage = _support_coverage(gt_root)
    failures = _failure_rows(edge_root, gt_root, local_ap_root, anchor_only_root)

    local_subset_gate_pass = bool(local_summary.get("phase_r5_4_diag_pass", False))
    full_dev_allowed = bool(local_subset_gate_pass and local_summary.get("fully_passing_r5_feature_variants"))
    history_allowed = False
    method_progress_allowed = bool(full_dev_allowed)
    diagnostic_progress = bool(_num(coverage.get("S_support_hit_rate"), 0.0) >= 0.95 and not method_progress_allowed)

    decision_rows = [
        _claim_row(
            "support_weighted_primitive_affinity_local_progress",
            method_progress_allowed,
            f"R5-4 fully_passing={local_summary.get('fully_passing_r5_feature_variants', [])}",
            "Not allowed: no support-weighted feature variant passed subset MV_AP/AP50 replay gates.",
        ),
        _claim_row(
            "full_dev_local_progress",
            full_dev_allowed,
            "Phase R5-5 was not entered.",
            "Not allowed until at least one R5-4 subset variant passes all local gates.",
        ),
        _claim_row(
            "holdout_ready",
            False,
            "No full-dev pass and no holdout run.",
            "Not allowed.",
        ),
        _claim_row(
            "local2history_progress",
            history_allowed,
            "S4 had history control-bias No-Go; R5 local gate failed before history.",
            "Not allowed; R5-6 was not run.",
        ),
        _claim_row(
            "diagnostic_progress",
            diagnostic_progress,
            f"S_support_hit_rate={coverage.get('S_support_hit_rate', '')}; R5-3 complete={gt_summary.get('phase_r5_3_diag_complete', False)}",
            (
                "Allowed only as diagnostic: support coverage exists but did not translate into local AP gate pass."
                if diagnostic_progress
                else "Not allowed as plan-defined diagnostic-progress: strict support GT coverage is low or false-connection diagnostic is nonzero."
            ),
        ),
    ]

    summary = {
        "schema_version": "stream4d_v103_supp_r5_final_decision_summary_v1",
        "phase_id": PHASE_ID,
        "decision": "NO_GO_SUPPORT_WEIGHTED_LOCAL_AP_GATE_FAILED",
        "overall_status": "diagnostic_progress_support_coverage_not_used_no_method_claim" if diagnostic_progress else "support_not_ready_no_method_claim",
        "phase_r5_0_pass": bool(fact_summary.get("phase_r5_0_pass", False)),
        "phase_r5_1_pass": bool(feature_summary.get("phase_r5_1_pass", False)),
        "phase_r5_2_pass": bool(edge_summary.get("phase_r5_2_pass", False)),
        "phase_r5_3_diag_complete": bool(gt_summary.get("phase_r5_3_diag_complete", False)),
        "phase_r5_4_diag_pass": bool(local_summary.get("phase_r5_4_diag_pass", False)),
        "anchor_only_phase_r5_4_diag_pass": bool(anchor_summary.get("phase_r5_4_diag_pass", False)),
        "local_subset_gate_pass": local_subset_gate_pass,
        "full_dev_allowed": full_dev_allowed,
        "history_allowed": history_allowed,
        "best_observed_local_ap_row": best,
        "support_coverage_all": coverage,
        "failure_count": len(failures),
        "outputs": {
            "summary": _rel(out / "summary.json"),
            "decision_rows": _rel(out / "decision_rows.csv"),
            "method_claim_boundary": _rel(out / "method_claim_boundary.md"),
            "failure_taxonomy_rows": _rel(out / "failure_taxonomy_rows.csv"),
        },
        "truthfulness_note": "R5 final decision is a gate aggregation. It does not rerun AP and does not use GT diagnostics for threshold selection.",
        "uses_gt_for_prediction": False,
        "uses_gt_for_eval": True,
        "uses_future": False,
        "runtime_sec": time.time() - t0,
    }

    _write_csv(out / "decision_rows.csv", decision_rows)
    _write_csv(out / "failure_taxonomy_rows.csv", failures)
    (out / "method_claim_boundary.md").write_text(_markdown(summary, decision_rows, failures), encoding="utf-8")
    _write_json(out / "summary.json", summary)
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default=str(DEFAULT_OUT))
    parser.add_argument("--fact-lock-root", default=str(DEFAULT_FACT_LOCK_ROOT))
    parser.add_argument("--feature-root", default=str(DEFAULT_FEATURE_ROOT))
    parser.add_argument("--edge-root", default=str(DEFAULT_EDGE_ROOT))
    parser.add_argument("--gt-root", default=str(DEFAULT_GT_ROOT))
    parser.add_argument("--local-ap-root", default=str(DEFAULT_LOCAL_AP_ROOT))
    parser.add_argument("--anchor-only-root", default=str(DEFAULT_ANCHOR_ONLY_ROOT))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    build(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
