#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
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
PHASE_ID = "v103_r7_final_casebook"
DEFAULT_OUT = AUDIT_ROOT / "v103_r7_final_casebook"


INPUT_SUMMARIES = {
    "R7-0": AUDIT_ROOT / "v103_r7_phase0_d4rt_only_fact_lock/summary.json",
    "R7-1": AUDIT_ROOT / "v103_r7_phase1_edge_attribution_exact/summary.json",
    "R7-3": AUDIT_ROOT / "v103_r7_phase3_anchor_confirmed_support/summary.json",
    "R7-4": AUDIT_ROOT / "v103_r7_phase4_skeleton_confirmed_support/summary.json",
    "R7-5": AUDIT_ROOT / "v103_r7_phase5_anchor_seeded_support_propagation/summary.json",
    "R7-7": AUDIT_ROOT / "v103_r7_phase7_local_eval_and_3d_inconsistency/summary.json",
}
R7_7_METRICS = AUDIT_ROOT / "v103_r7_phase7_local_eval_and_3d_inconsistency/metric_rows.csv"


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


def _sha256(path: Path) -> str:
    if not path.exists() or not path.is_file():
        return ""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _artifact_row(role: str, path: Path, source_phase: str = "") -> dict[str, Any]:
    exists = path.exists()
    row_count: int | str = ""
    if exists and path.is_file() and path.suffix.lower() in {".csv", ".parquet"}:
        try:
            row_count = max(sum(1 for _ in path.open("r", encoding="utf-8")) - 1, 0) if path.suffix.lower() == ".csv" else int(pd.read_parquet(path).shape[0])
        except Exception:
            row_count = ""
    return {
        "schema_version": "stream4d_v103_r7_final_artifact_row_v1",
        "phase_id": PHASE_ID,
        "source_phase": source_phase,
        "artifact_role": role,
        "path": _rel(path),
        "exists": exists,
        "size_bytes": path.stat().st_size if exists and path.is_file() else "",
        "row_count": row_count,
        "sha256": _sha256(path),
        "uses_gt_for_prediction": False,
        "uses_future": False,
    }


def _num(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        out = float(value)
        return out if np.isfinite(out) else default
    except Exception:
        return default


def _clean(value: Any) -> Any:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    return value


def build(args: argparse.Namespace) -> dict[str, Any]:
    t0 = time.time()
    out = Path(args.output_root)
    if not out.is_absolute():
        out = STREAM3D_ROOT / out
    if out.exists() and any(out.iterdir()) and not args.force:
        raise SystemExit(f"output root already exists and is non-empty; pass --force: {_rel(out)}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "last_command.txt").write_text(" ".join([sys.executable, str(Path(__file__)), *sys.argv[1:]]) + "\n", encoding="utf-8")

    summaries = {phase: _read_json(path) for phase, path in INPUT_SUMMARIES.items()}
    artifact_rows = [_artifact_row("summary", path, phase) for phase, path in INPUT_SUMMARIES.items()]
    artifact_rows.append(_artifact_row("r7_7_metric_rows", R7_7_METRICS, "R7-7"))
    artifact_rows.append(_artifact_row("last_command", out / "last_command.txt", PHASE_ID))

    metric_rows = pd.read_csv(R7_7_METRICS) if R7_7_METRICS.exists() else pd.DataFrame()
    real_rows = metric_rows[metric_rows["control_role"].astype(str).eq("real")] if not metric_rows.empty else pd.DataFrame()
    best = real_rows.sort_values(["MV_AP_window", "MV_AP50_window"], ascending=False).iloc[0].to_dict() if not real_rows.empty else {}
    best_control_delta = _num(best.get("real_minus_best_control_MV_AP_window"), 0.0)
    r7_0 = summaries.get("R7-0", {})
    r7_7 = summaries.get("R7-7", {})
    phase_passes = {phase: bool(summary.get("phase_pass", False)) for phase, summary in summaries.items()}
    local_claim_allowed = False
    scene_claim_allowed = False
    holdout_allowed = False
    decision = "NO_GO_R7_SUPPORT_NOT_OBJECT_SPECIFIC"
    primary_blocker = "R7-3/R7-4/R7-5 failed subset promotion; R7-7 found no local or history candidate."
    secondary_blockers = [
        "R7-1 exact attribution missing six leave-one-family-out references.",
        "R7-4 best MV_AP_window signal did not meet threshold and had diff-GT false connection rate 0.5555555555555556.",
        "R7-5 real one-hop propagation produced zero propagated edges.",
        "R7-2/R7-6 were not run in the current main ladder and are not fabricated.",
    ]
    next_action = "Do not run R7-8/R7-9/full-dev. Either implement missing exact R7-1 custom intervention references or start a new plan."

    gate_rows = [
        {
            "schema_version": "stream4d_v103_r7_final_gate_row_v1",
            "phase_id": PHASE_ID,
            "gate_name": "d4rt_only_boundary",
            "pass": not any(bool(s.get("DA3_USED", False)) or bool(s.get("GS_USED", False)) for s in summaries.values()),
            "observed": {phase: {"DA3_USED": summary.get("DA3_USED", ""), "GS_USED": summary.get("GS_USED", "")} for phase, summary in summaries.items()},
            "required": "DA3_USED=false and GS_USED=false for all R7 main-path phases",
            "uses_gt_for_prediction": False,
            "uses_future": False,
        },
        {
            "schema_version": "stream4d_v103_r7_final_gate_row_v1",
            "phase_id": PHASE_ID,
            "gate_name": "local_candidate_available",
            "pass": local_claim_allowed,
            "observed": phase_passes,
            "required": "at least one R7 local subset candidate pass",
            "uses_gt_for_prediction": False,
            "uses_future": False,
        },
        {
            "schema_version": "stream4d_v103_r7_final_gate_row_v1",
            "phase_id": PHASE_ID,
            "gate_name": "history_candidate_available",
            "pass": bool(summaries.get("R7-4", {}).get("history_candidate", False)),
            "observed": summaries.get("R7-4", {}).get("history_candidate", ""),
            "required": "local or inconsistency candidate before R7-8",
            "uses_gt_for_prediction": False,
            "uses_future": False,
        },
        {
            "schema_version": "stream4d_v103_r7_final_gate_row_v1",
            "phase_id": PHASE_ID,
            "gate_name": "holdout_allowed",
            "pass": holdout_allowed,
            "observed": False,
            "required": "full-dev frozen config must pass before holdout",
            "uses_gt_for_prediction": False,
            "uses_future": False,
        },
    ]
    failure_rows = [
        {
            "schema_version": "stream4d_v103_r7_final_failure_row_v1",
            "phase_id": PHASE_ID,
            "failure_id": "NO_GO_R7_SUPPORT_NOT_OBJECT_SPECIFIC",
            "severity": "blocking",
            "detail": primary_blocker,
            "repair_direction": next_action,
            "uses_gt_for_prediction": False,
            "uses_future": False,
        }
    ]
    final_metric_rows = [
        {
            "schema_version": "stream4d_v103_r7_final_metric_row_v1",
            "phase_id": PHASE_ID,
            "best_local_variant_id": best.get("variant_id", ""),
            "best_local_variant_family": best.get("variant_family", ""),
            "best_local_MV_AP_window": _clean(best.get("MV_AP_window", "")),
            "best_local_MV_AP50_window": _clean(best.get("MV_AP50_window", "")),
            "real_minus_best_control_local": best_control_delta,
            "accepted_S_only_edge_count": _clean(best.get("accepted_S_only_edge_count", "")),
            "accepted_diff_gt_edge_count_diagnostic": _clean(best.get("accepted_diff_gt_edge_count_diagnostic", "")),
            "GT_fragment_count_ge2_rate_delta": "",
            "same_GT_connection_rate_delta": "",
            "uses_gt_for_prediction": False,
            "uses_future": False,
        }
    ]
    _write_csv(out / "artifact_rows.csv", artifact_rows)
    _write_csv(out / "gate_rows.csv", gate_rows)
    _write_csv(out / "failure_rows.csv", failure_rows)
    _write_csv(out / "metric_rows.csv", final_metric_rows)
    _write_csv(out / "control_rows.csv", [])
    _write_csv(out / "variant_rows.csv", real_rows.to_dict("records") if not real_rows.empty else [])
    summary = {
        "schema_version": "stream4d_v103_r7_final_summary_v1",
        "phase": "R7-final",
        "phase_id": PHASE_ID,
        "decision": decision,
        "local_claim_allowed": local_claim_allowed,
        "scene_claim_allowed": scene_claim_allowed,
        "holdout_allowed": holdout_allowed,
        "best_local_variant_id": best.get("variant_id", ""),
        "best_local_MV_AP_window": _clean(best.get("MV_AP_window", "")),
        "best_local_MV_AP50_window": _clean(best.get("MV_AP50_window", "")),
        "best_scene_variant_id": "",
        "best_MV_AP_scene": "",
        "best_MV_AP50_scene": "",
        "real_minus_best_control_local": best_control_delta,
        "real_minus_best_control_scene": "",
        "accepted_S_only_edge_count": _clean(best.get("accepted_S_only_edge_count", "")),
        "accepted_diff_gt_edge_count_diagnostic": _clean(best.get("accepted_diff_gt_edge_count_diagnostic", "")),
        "GT_fragment_count_ge2_rate_delta": "",
        "same_GT_connection_rate_delta": "",
        "objects_crossing_multiple_chunks": "",
        "history_control_bias_status": "not_run_no_r7_8_candidate",
        "DA3_USED": False,
        "uses_gt_for_prediction": False,
        "uses_future": False,
        "primary_blocker": primary_blocker,
        "secondary_blockers": secondary_blockers,
        "next_action": next_action,
        "r7_0_decision": r7_0.get("decision", ""),
        "r7_7_decision": r7_7.get("decision", ""),
        "runtime_sec": time.time() - t0,
        "outputs": {
            "summary": _rel(out / "summary.json"),
            "gate_rows": _rel(out / "gate_rows.csv"),
            "failure_rows": _rel(out / "failure_rows.csv"),
            "metric_rows": _rel(out / "metric_rows.csv"),
            "artifact_rows": _rel(out / "artifact_rows.csv"),
        },
        "truthfulness_note": "Final decision is based on generated R7-0/R7-1/R7-3/R7-4/R7-5/R7-7 artifacts only. R7-8/R7-9/holdout were not run because no subset candidate passed.",
    }
    _write_json(out / "summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build v103 R7 final casebook.")
    parser.add_argument("--output-root", default=str(DEFAULT_OUT))
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    summary = build(parse_args())
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
