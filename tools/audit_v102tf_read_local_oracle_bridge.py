#!/usr/bin/env python3
"""Audit v102 READ-local oracle bridge from legacy H2/L07 evidence.

This audit reuses artifact-backed v97/H2 READ evidence as a Stage3 diagnostic
oracle input. It explicitly separates legacy aggregate local evidence from
v102 case-aligned promotion evidence, and records global/full-sequence harm
evidence required by the v102 plan when READ helps locally but is not a full
method.
"""

from __future__ import annotations

import csv
import json
import math
import shutil
from pathlib import Path
from typing import Any


ROOT = Path("results/acl2_v102tf_drift_source_autopsy_semantic_oracle_memory_control")
STAGE2_BASE_ROWS = ROOT / "stage2_base_case_selection/base_case_rows.csv"
OUT = ROOT / "stage3_semantic_oracle_upper_bound"

V97_ROOT = Path("results/acl2_v97tf_semantic_scale_evidence_gauge_safe_memory_control")
H2_SUMMARY = V97_ROOT / "trackH2_l07_component_decomposition/summary.json"
H2_COMPONENT_ROWS = V97_ROOT / "trackH2_l07_component_decomposition/component_rows.csv"
TRACKI_ROOT = V97_ROOT / "trackI_scale_gauge_evidence_observatory_v2"
TRACKI_ACTIVE_INACTIVE = TRACKI_ROOT / "active_inactive_tradeoff_rows.csv"
TRACKI_FULL_SEQUENCE = TRACKI_ROOT / "full_sequence_gate_rows.csv"
TRACKI_LATENT_SHIFT = TRACKI_ROOT / "latent_gauge_shift_rows.csv"

OUT_ROWS = OUT / "read_local_oracle_bridge_rows.csv"
OUT_SUMMARY = OUT / "read_local_oracle_bridge_summary.json"
OUT_REPORT = OUT / "read_local_oracle_bridge_report.md"
OUT_HELP_HARM = OUT / "read_local_help_global_harm_report.md"
OUT_ACTIVE_INACTIVE = OUT / "read_local_active_inactive_global_tradeoff.csv"
OUT_SCALE_YAW = OUT / "read_local_global_sim3_scale_yaw_shift.csv"


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {"value": payload}


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    if not keys:
        keys = ["empty"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: csv_value(row.get(key)) for key in keys})


def csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, float) and not math.isfinite(value):
        return ""
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(jsonable(value), sort_keys=True)
    return value


def jsonable(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def f(value: Any, default: float = math.nan) -> float:
    if value is None or value == "":
        return default
    try:
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def median(values: list[float]) -> float:
    finite = sorted(value for value in values if math.isfinite(value))
    if not finite:
        return math.nan
    mid = len(finite) // 2
    if len(finite) % 2:
        return finite[mid]
    return 0.5 * (finite[mid - 1] + finite[mid])


def boolish(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def copy_csv(src: Path, dst: Path) -> int:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not src.is_file():
        write_rows(dst, [{"source": src.as_posix(), "available": False, "missing_reason": "source_file_missing"}])
        return 0
    shutil.copy2(src, dst)
    return len(read_rows(dst))


def source_pilot_per_case_path(best_component: dict[str, Any]) -> Path:
    root = Path(str(best_component.get("source_pilot_root", "")))
    return root / "per_case_candidate_comparison.csv"


def metric_rows_by_case(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    out: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        case_id = row.get("case_id", "")
        if case_id:
            out.setdefault(case_id, []).append(row)
    return out


def v102_role(row: dict[str, str]) -> str:
    if row.get("selection_label") == "READ_LOCAL_SCALE" or row.get("base_case_group") == "R":
        return "v102_strict_read_local_base"
    labels = row.get("drift_source_labels", "")
    if "READ_LOCAL_SCALE" in labels or row.get("primary_drift_source") == "READ_LOCAL_SCALE":
        return "v102_mixed_read_local_evidence"
    if row.get("primary_drift_source") == "SAFE_GOOD" or row.get("base_case_group") == "G":
        return "v102_good_control"
    return "v102_other_base_case"


def summarize_metric(rows: list[dict[str, str]], metric: str) -> dict[str, float]:
    scoped = [row for row in rows if row.get("metric") == metric]
    return {
        "count": len(scoped),
        "median_improvement_vs_baseline": median([f(row.get("candidate_improvement_vs_baseline")) for row in scoped]),
        "median_min_margin_vs_controls": median([f(row.get("candidate_min_margin_vs_controls")) for row in scoped]),
        "pass_case_count": sum(
            1
            for row in scoped
            if f(row.get("candidate_improvement_vs_baseline")) >= 0.05
            and f(row.get("candidate_min_margin_vs_controls")) >= 0.05
        ),
    }


def make_scale_yaw_rows(full_rows: list[dict[str, str]], latent_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    latent = latent_rows[0] if latent_rows else {}
    scale_available = boolish(latent.get("available"))
    missing = latent.get("missing_reason", "latent shift rows unavailable")
    out: list[dict[str, Any]] = []
    for row in full_rows:
        out.append(
            {
                "source_json": row.get("source_json", ""),
                "candidate": row.get("candidate", ""),
                "strict_full_gate_pass": row.get("strict_full_gate_pass", ""),
                "delta_aligned_ate_rmse_m": row.get("delta_aligned_ate_rmse_m", ""),
                "delta_final_error_m": row.get("delta_final_error_m", ""),
                "delta_yaw_rmse_deg": row.get("delta_yaw_rmse_deg", ""),
                "delta_error_slope_m_per_100f": row.get("delta_error_slope_m_per_100f", ""),
                "rolling_worse_fraction_max": row.get("rolling_worse_fraction_max", ""),
                "global_scale_shift_available": scale_available,
                "global_scale_shift_missing_reason": "" if scale_available else missing,
                "claim_level": "global_yaw_proxy_and_scale_missing_diagnostic",
            }
        )
    return out


def main() -> int:
    h2_summary = read_json(H2_SUMMARY)
    best = h2_summary.get("best_passing_component")
    best = best if isinstance(best, dict) else {}
    source_rows = read_rows(source_pilot_per_case_path(best))
    h2_components = read_rows(H2_COMPONENT_ROWS)
    stage2_rows = read_rows(STAGE2_BASE_ROWS)
    full_rows = read_rows(TRACKI_FULL_SEQUENCE)
    latent_rows = read_rows(TRACKI_LATENT_SHIFT)
    stage2_by_case = {row.get("case_id", ""): row for row in stage2_rows if row.get("case_id")}
    source_by_case = metric_rows_by_case(source_rows)
    source_cases = sorted(source_by_case)
    stage2_overlap = sorted(set(stage2_by_case) & set(source_by_case))
    strict_read_cases = [
        row.get("case_id", "")
        for row in stage2_rows
        if row.get("selection_label") == "READ_LOCAL_SCALE" or row.get("base_case_group") == "R"
    ]
    broad_read_cases = [
        row.get("case_id", "")
        for row in stage2_rows
        if "READ_LOCAL_SCALE" in row.get("drift_source_labels", "")
        or row.get("primary_drift_source") == "READ_LOCAL_SCALE"
        or row.get("target_memory_body") == "READ"
    ]
    good_control_cases = [
        row.get("case_id", "")
        for row in stage2_rows
        if row.get("base_case_group") == "G" or row.get("primary_drift_source") == "SAFE_GOOD"
    ]
    strict_read_overlap = sorted(set(strict_read_cases) & set(source_cases))
    broad_read_overlap = sorted(set(broad_read_cases) & set(source_cases))
    good_control_overlap = sorted(set(good_control_cases) & set(source_cases))
    rows: list[dict[str, Any]] = []
    for case_id in source_cases:
        v102 = stage2_by_case.get(case_id, {})
        for metric_row in source_by_case[case_id]:
            rows.append(
                {
                    "case_id": case_id,
                    "source_bucket": metric_row.get("bucket", ""),
                    "metric": metric_row.get("metric", ""),
                    "baseline": metric_row.get("baseline", ""),
                    "candidate": metric_row.get("candidate", ""),
                    "random_same_mass": metric_row.get("random_same_mass", ""),
                    "semantic_rotation": metric_row.get("semantic_rotation", ""),
                    "candidate_improvement_vs_baseline": metric_row.get("candidate_improvement_vs_baseline", ""),
                    "candidate_min_margin_vs_controls": metric_row.get("candidate_min_margin_vs_controls", ""),
                    "candidate_margin_vs_random": metric_row.get("candidate_margin_vs_random", ""),
                    "candidate_margin_vs_semantic_rotation": metric_row.get("candidate_margin_vs_semantic_rotation", ""),
                    "in_v102_stage2_base_cases": bool(v102),
                    "v102_role": v102_role(v102) if v102 else "not_in_v102_stage2_base_cases",
                    "v102_seq": v102.get("seq", metric_row.get("seq", "")),
                    "v102_selection_label": v102.get("selection_label", ""),
                    "v102_base_case_group": v102.get("base_case_group", ""),
                    "v102_label_original": v102.get("label_original", ""),
                    "v102_failure_type_original": v102.get("failure_type_original", ""),
                    "v102_target_taxonomy_v101": v102.get("target_taxonomy_v101", ""),
                    "v102_primary_drift_source": v102.get("primary_drift_source", ""),
                    "v102_drift_source_labels": v102.get("drift_source_labels", ""),
                    "claim_level": "read_local_legacy_bridge_case_diagnostic",
                }
            )
    strict_read_metric_rows = [
        row for row in rows if row["case_id"] in strict_read_overlap and row["metric"] == "scale_cv_head_mid_tail_pose_sim3"
    ]
    good_metric_rows = [
        row for row in rows if row["case_id"] in good_control_overlap and row["metric"] == "scale_cv_head_mid_tail_pose_sim3"
    ]
    strict_read_summary = summarize_metric(strict_read_metric_rows, "scale_cv_head_mid_tail_pose_sim3")
    good_summary = summarize_metric(good_metric_rows, "scale_cv_head_mid_tail_pose_sim3")
    legacy_upper_bound_pass = (
        boolish(h2_summary.get("gate_pass"))
        and f(best.get("bad_L2_improvement")) >= 0.05
        and f(best.get("good_worsen")) <= 0.02
        and f(best.get("candidate_min_margin_vs_required_controls")) >= 0.05
    )
    v102_case_aligned_pass = (
        len(strict_read_overlap) >= 3
        and len(good_control_overlap) >= 2
        and strict_read_summary["median_improvement_vs_baseline"] >= 0.05
        and strict_read_summary["median_min_margin_vs_controls"] >= 0.05
        and good_summary["median_improvement_vs_baseline"] >= -0.02
    )
    active_inactive_count = copy_csv(TRACKI_ACTIVE_INACTIVE, OUT_ACTIVE_INACTIVE)
    scale_yaw_rows = make_scale_yaw_rows(full_rows, latent_rows)
    write_rows(OUT_SCALE_YAW, scale_yaw_rows)
    strict_full_pass_count = sum(1 for row in full_rows if boolish(row.get("strict_full_gate_pass")))
    full_sequence_no_go = bool(full_rows) and strict_full_pass_count == 0
    summary = {
        "schema": "acl2_v102_read_local_oracle_bridge_v1",
        "diagnostic_only": True,
        "runtime_action_allowed": False,
        "legacy_h2_gate_pass": boolish(h2_summary.get("gate_pass")),
        "legacy_h2_local_L2_mechanism_exists": boolish(h2_summary.get("local_L2_mechanism_exists")),
        "legacy_h2_best_candidate": best.get("candidate", ""),
        "legacy_h2_best_bad_L2_improvement": f(best.get("bad_L2_improvement")),
        "legacy_h2_best_good_worsen": f(best.get("good_worsen")),
        "legacy_h2_best_control_margin": f(best.get("candidate_min_margin_vs_required_controls")),
        "legacy_read_local_upper_bound_pass": legacy_upper_bound_pass,
        "source_pilot_case_count": len(source_cases),
        "source_pilot_metric_row_count": len(source_rows),
        "h2_component_row_count": len(h2_components),
        "v102_stage2_case_count": len(stage2_rows),
        "v102_strict_read_local_case_count": len(strict_read_cases),
        "v102_strict_read_local_sequence_count": len({stage2_by_case[case]["seq"] for case in strict_read_cases if case in stage2_by_case}),
        "v102_broad_read_local_case_count": len(broad_read_cases),
        "v102_broad_read_local_sequence_count": len({stage2_by_case[case]["seq"] for case in broad_read_cases if case in stage2_by_case}),
        "v102_good_control_case_count": len(good_control_cases),
        "source_v102_stage2_overlap_count": len(stage2_overlap),
        "source_v102_stage2_overlap_cases": ";".join(stage2_overlap),
        "v102_strict_read_local_source_overlap_count": len(strict_read_overlap),
        "v102_strict_read_local_source_overlap_cases": ";".join(strict_read_overlap),
        "v102_broad_read_local_source_overlap_count": len(broad_read_overlap),
        "v102_broad_read_local_source_overlap_cases": ";".join(broad_read_overlap),
        "v102_good_control_source_overlap_count": len(good_control_overlap),
        "v102_good_control_source_overlap_cases": ";".join(good_control_overlap),
        "v102_case_aligned_scale_metric_median_improvement": strict_read_summary["median_improvement_vs_baseline"],
        "v102_case_aligned_scale_metric_median_control_margin": strict_read_summary["median_min_margin_vs_controls"],
        "v102_case_aligned_scale_metric_pass_case_count": strict_read_summary["pass_case_count"],
        "v102_case_aligned_good_control_median_improvement": good_summary["median_improvement_vs_baseline"],
        "v102_case_aligned_read_local_oracle_pass": v102_case_aligned_pass,
        "full_sequence_rows_count": len(full_rows),
        "full_sequence_strict_pass_count": strict_full_pass_count,
        "full_sequence_no_go": full_sequence_no_go,
        "active_inactive_tradeoff_rows_copied": active_inactive_count,
        "global_scale_shift_available": boolish(latent_rows[0].get("available")) if latent_rows else False,
        "global_scale_shift_missing_reason": latent_rows[0].get("missing_reason", "latent shift rows unavailable") if latent_rows else "latent shift rows unavailable",
        "stage3_strict_semantic_oracle_pass": False,
        "stage4_allowed": False,
        "blocker": (
            "Legacy H2 READ local aggregate upper-bound passes, but v102 case-aligned evidence is insufficient: "
            "strict READ_LOCAL overlap is below the required case/control coverage, and full-sequence READ remains No-Go."
        ),
        "outputs": {
            "rows": OUT_ROWS.as_posix(),
            "report": OUT_REPORT.as_posix(),
            "read_local_help_global_harm_report": OUT_HELP_HARM.as_posix(),
            "active_inactive_global_tradeoff": OUT_ACTIVE_INACTIVE.as_posix(),
            "global_sim3_scale_yaw_shift": OUT_SCALE_YAW.as_posix(),
        },
    }
    write_rows(OUT_ROWS, rows)
    write_json(OUT_SUMMARY, summary)
    write_text(
        OUT_REPORT,
        "\n".join(
            [
                "# READ Local Oracle Bridge",
                "",
                "This audit bridges legacy v97/H2 READ local evidence into v102 Stage3 without promoting it to runtime.",
                "",
                "## Summary",
                "",
                f"- legacy_h2_gate_pass: {summary['legacy_h2_gate_pass']}",
                f"- legacy_h2_best_candidate: {summary['legacy_h2_best_candidate']}",
                f"- legacy_h2_best_bad_L2_improvement: {summary['legacy_h2_best_bad_L2_improvement']}",
                f"- legacy_h2_best_good_worsen: {summary['legacy_h2_best_good_worsen']}",
                f"- legacy_h2_best_control_margin: {summary['legacy_h2_best_control_margin']}",
                f"- legacy_read_local_upper_bound_pass: {summary['legacy_read_local_upper_bound_pass']}",
                f"- v102_strict_read_local_case_count: {summary['v102_strict_read_local_case_count']}",
                f"- v102_strict_read_local_source_overlap_count: {summary['v102_strict_read_local_source_overlap_count']}",
                f"- v102_strict_read_local_source_overlap_cases: {summary['v102_strict_read_local_source_overlap_cases']}",
                f"- v102_good_control_source_overlap_count: {summary['v102_good_control_source_overlap_count']}",
                f"- v102_case_aligned_read_local_oracle_pass: {summary['v102_case_aligned_read_local_oracle_pass']}",
                f"- full_sequence_strict_pass_count: {summary['full_sequence_strict_pass_count']}",
                f"- full_sequence_no_go: {summary['full_sequence_no_go']}",
                "",
                "## Conclusion",
                "",
                summary["blocker"],
            ]
        ),
    )
    write_text(
        OUT_HELP_HARM,
        "\n".join(
            [
                "# READ Local Help / Global Harm Report",
                "",
                "Legacy H2 shows a local READ upper-bound signal, but Track I full-sequence rows do not pass strict full gates.",
                "",
                f"- local_upper_bound_pass: {legacy_upper_bound_pass}",
                f"- full_sequence_rows_count: {len(full_rows)}",
                f"- full_sequence_strict_pass_count: {strict_full_pass_count}",
                f"- full_sequence_no_go: {full_sequence_no_go}",
                f"- active_inactive_tradeoff_rows: {active_inactive_count}",
                f"- global_scale_shift_available: {summary['global_scale_shift_available']}",
                f"- global_scale_shift_missing_reason: {summary['global_scale_shift_missing_reason']}",
                "",
                "Interpretation:",
                "",
                "READ may remain a provider/local oracle, but current artifacts do not support promoting READ to a full runtime memory-control method. "
                "The required global scale/yaw analysis is limited by missing saved stable-anchor latent embeddings; yaw/final-error/rolling proxies are preserved in the CSV outputs.",
            ]
        ),
    )
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
