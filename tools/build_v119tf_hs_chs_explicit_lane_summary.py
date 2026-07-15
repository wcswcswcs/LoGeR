#!/usr/bin/env python3
"""Summarize v119TF HorizonStream C-HS explicit-lane smoke/control rows."""

from __future__ import annotations

import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/acl2_v119tf_semantic_addressable_geometry_carrier_routing_carrier_aware_augmented"
RUN_ROOT = OUT / "stage4_hs_dhs_liveness_smoke"
INPUT_RESULTS = [
    RUN_ROOT / "dhs_liveness_smoke_max12_global_mrt_chs2l_smoke_run_results.csv",
    RUN_ROOT / "dhs_liveness_smoke_max12_global_mrt_chs3l_chss_smoke_run_results.csv",
]
SUMMARY_JSON = RUN_ROOT / "dhs_chs_explicit_lane_smoke_summary.json"
SUMMARY_CSV = RUN_ROOT / "dhs_chs_explicit_lane_smoke_summary.csv"


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def finite_float(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def as_float(value: Any) -> float | None:
    return float(value) if finite_float(value) else None


def truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def load_lane_audit(row: dict[str, str]) -> dict[str, Any]:
    root = ROOT / row.get("action_audit_root", "")
    path = root / "hs_chs_lane_action_rows.csv"
    audit_rows = read_csv(path)
    changed = [r for r in audit_rows if truthy(r.get("changed_state"))]
    lane_forms = sorted({r.get("lane_form", "") for r in audit_rows if r.get("lane_form")})
    calibration_modes = sorted({r.get("calibration_mode", "") for r in audit_rows if r.get("calibration_mode")})
    assignments = {
        "transient": [float(r["transient_assignment"]) for r in audit_rows if finite_float(r.get("transient_assignment"))],
        "persistent": [float(r["persistent_assignment"]) for r in audit_rows if finite_float(r.get("persistent_assignment"))],
        "metric": [float(r["metric_assignment"]) for r in audit_rows if finite_float(r.get("metric_assignment"))],
    }
    return {
        "audit_path": rel(path) if path.is_file() else "",
        "lane_audit_row_count": len(audit_rows),
        "lane_changed_state_row_count": len(changed),
        "lane_forms": lane_forms,
        "calibration_modes": calibration_modes,
        "transient_assignment_mean": mean(assignments["transient"]),
        "persistent_assignment_mean": mean(assignments["persistent"]),
        "metric_assignment_mean": mean(assignments["metric"]),
        "first_chunk_no_prior_count": sum(1 for r in audit_rows if truthy(r.get("first_chunk_no_prior"))),
        "stored_lane_count_max": max(
            [int(float(r["stored_lane_count"])) for r in audit_rows if finite_float(r.get("stored_lane_count"))],
            default=0,
        ),
    }


def case_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for input_path in INPUT_RESULTS:
        for row in read_csv(input_path):
            if not row.get("branch", "").startswith("C-HS"):
                continue
            audit = load_lane_audit(row)
            ate = as_float(row.get("ate_rmse"))
            enriched = {
                "schema": "acl2_v119tf_hs_chs_explicit_lane_case_row_v1",
                "source_results_csv": rel(input_path),
                "case_id": row.get("case_id"),
                "branch": row.get("branch"),
                "role": row.get("role"),
                "seq": row.get("seq"),
                "max_frames": 12,
                "trace_profile": row.get("trace_profile"),
                "action": row.get("action"),
                "control": row.get("control"),
                "ate_rmse": ate,
                "returncode": row.get("returncode"),
                "liveness_pass": truthy(row.get("liveness_pass")),
                "expected_audit_rows": int(float(row.get("expected_audit_rows") or 0)),
                "trace_total_rows": int(float(row.get("trace_total_rows") or 0)),
                **audit,
            }
            rows.append(enriched)
    return rows


def summarize_branch(branch: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    branch_rows = [row for row in rows if row["branch"] == branch]
    candidates = [row for row in branch_rows if "control" not in str(row.get("role", ""))]
    controls = [row for row in branch_rows if "control" in str(row.get("role", ""))]
    finite_candidates = [row for row in candidates if row.get("ate_rmse") is not None]
    finite_controls = [row for row in controls if row.get("ate_rmse") is not None]
    best_candidate = min(finite_candidates, key=lambda r: float(r["ate_rmse"])) if finite_candidates else None
    best_control = min(finite_controls, key=lambda r: float(r["ate_rmse"])) if finite_controls else None
    candidate_beats_all_controls = bool(
        best_candidate is not None
        and finite_controls
        and all(float(best_candidate["ate_rmse"]) < float(row["ate_rmse"]) for row in finite_controls)
    )
    all_liveness = bool(branch_rows and all(bool(row.get("liveness_pass")) for row in branch_rows))
    control_gap = None
    if best_candidate is not None and best_control is not None:
        control_gap = float(best_control["ate_rmse"]) - float(best_candidate["ate_rmse"])
    if not all_liveness:
        status = "SMOKE_INCOMPLETE_OR_LIVENESS_FAIL"
        blocker = "one_or_more_cases_failed_liveness"
    elif candidate_beats_all_controls:
        status = "SMOKE_COMPLETE_CANDIDATE_BEATS_CONTROLS_FULL_VALIDATION_PENDING"
        blocker = ""
    else:
        status = "SMOKE_COMPLETE_NO_GO_SEMANTIC_SPECIFICITY_OR_CAPACITY_CONTROL_FAIL"
        blocker = (
            f"best_control={best_control['case_id']} ate={best_control['ate_rmse']} "
            f"beats_or_matches best_candidate={best_candidate['case_id']} ate={best_candidate['ate_rmse']}"
            if best_candidate and best_control
            else "candidate_or_control_metric_missing"
        )
    return {
        "schema": "acl2_v119tf_hs_chs_explicit_lane_branch_summary_v1",
        "branch": branch,
        "job_count": len(branch_rows),
        "candidate_count": len(candidates),
        "control_count": len(controls),
        "all_liveness_pass": all_liveness,
        "best_candidate_case": best_candidate["case_id"] if best_candidate else "",
        "best_candidate_ate": best_candidate["ate_rmse"] if best_candidate else "",
        "best_control_case": best_control["case_id"] if best_control else "",
        "best_control_ate": best_control["ate_rmse"] if best_control else "",
        "best_control_minus_best_candidate_ate": control_gap,
        "candidate_beats_all_controls": candidate_beats_all_controls,
        "current_status": status,
        "primary_blocker": blocker,
        "truthfulness_boundary": (
            "seq00 max12/global_mrt explicit-lane smoke/control only; not full-sequence, "
            "not cross-sequence, not final v119 method validation"
        ),
    }


def main() -> int:
    generated_at = datetime.now(timezone.utc).isoformat()
    rows = case_rows()
    branches = sorted({row["branch"] for row in rows})
    summaries = [summarize_branch(branch, rows) for branch in branches]
    for row in summaries:
        row["generated_at_utc"] = generated_at
    payload = {
        "schema": "acl2_v119tf_hs_chs_explicit_lane_smoke_summary_v1",
        "generated_at_utc": generated_at,
        "input_results": [rel(path) for path in INPUT_RESULTS],
        "summary_csv": rel(SUMMARY_CSV),
        "case_count": len(rows),
        "branch_count": len(summaries),
        "all_jobs_liveness_pass": bool(rows and all(bool(row.get("liveness_pass")) for row in rows)),
        "terminal_pass": False,
        "completion_claim": "not_complete_explicit_lane_smoke_only",
        "branches": summaries,
        "cases": rows,
    }
    SUMMARY_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    write_csv(SUMMARY_CSV, summaries)
    print(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
