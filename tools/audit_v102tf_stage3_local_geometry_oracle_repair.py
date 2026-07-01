#!/usr/bin/env python3
"""Audit v102 Stage3 local-geometry oracle candidates on selected base cases.

This is a fail-forward diagnostic.  It joins the strict visual panels produced
for Stage2 (RGB/semantic/risk, trajectory error, and local point residual) and
tests simple threshold selectors per drift-source target.  Passing rows here do
not authorize runtime action because the audit is limited to selected base
cases and does not rerun full target-universe semantic-rotation/control probes.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any


ROOT = Path("results/acl2_v102tf_drift_source_autopsy_semantic_oracle_memory_control")
STAGE2 = ROOT / "stage2_base_case_selection"
STAGE3 = ROOT / "stage3_semantic_oracle_upper_bound"
BASE_ROWS = STAGE2 / "base_case_rows.csv"
RGB_ROWS = STAGE2 / "rgb_semantic_overlay_manifest.csv"
TRAJ_ROWS = STAGE2 / "trajectory_error_overlay_manifest.csv"
LOCAL_ROWS = STAGE2 / "local_point_residual_overlay_manifest.csv"
SUMMARY = STAGE3 / "stage3_local_geometry_oracle_repair_summary.json"
ROWS_OUT = STAGE3 / "stage3_local_geometry_oracle_repair_rows.csv"
CASE_OUT = STAGE3 / "stage3_local_geometry_oracle_case_rows.csv"
READINESS_OUT = STAGE3 / "stage3_full_control_rerun_readiness_rows.csv"
REPORT = STAGE3 / "stage3_local_geometry_oracle_repair_report.md"
V101_JOIN_ROWS = Path(
    "results/acl2_v101tf_semantic_anchor_state_estimation_scale_update_admission/"
    "final_decision/anchor_seed_lifecycle_geometry_observability_case_rows.csv"
)
V101_TARGET28_ROOT = Path(
    "results/acl2_v101tf_semantic_anchor_state_estimation_scale_update_admission/"
    "stage_c_seed_bridge_geometry_smoke_target28"
)
V102_REPAIR_TRACE_ROOT = STAGE2 / "local_point_sidecar_repair_traces"


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                keys.append(key)
                seen.add(key)
    if not keys:
        keys = ["empty"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: csv_value(row.get(key, "")) for key in keys})


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, float) and not math.isfinite(value):
        return ""
    if isinstance(value, (dict, list, tuple, set)):
        return json.dumps(jsonable(value), sort_keys=True)
    return value


def jsonable(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def f(value: Any, default: float = math.nan) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def by_case(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    return {str(row.get("case_id", "")): row for row in rows if row.get("case_id")}


def deterministic_rng(*parts: str) -> random.Random:
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()
    return random.Random(int(digest[:16], 16))


def mean(values: list[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return sum(finite) / len(finite) if finite else math.nan


def balanced_accuracy(recall: float, good_fpr: float) -> float:
    if not math.isfinite(recall) or not math.isfinite(good_fpr):
        return math.nan
    return 0.5 * (recall + (1.0 - good_fpr))


def labels_for(row: dict[str, Any]) -> dict[str, bool]:
    primary = str(row.get("primary_drift_source", ""))
    group = str(row.get("base_case_group", ""))
    label = str(row.get("label_original", ""))
    l3 = f(row.get("L3_handoff_transfer_penalty_proxy"))
    return {
        "swa_handoff": primary.startswith("SWA_HANDOFF"),
        "read_local": primary == "READ_LOCAL_SCALE",
        "non_good_target": group in {"R", "S", "L", "U"} or label == "bad",
        "high_l3_non_good": label != "good" and math.isfinite(l3) and l3 >= f(row.get("L3_q75"), 241.836179987942),
        "safe_good_control": group == "G" or primary == "SAFE_GOOD",
    }


def select(values: list[float], threshold: float, direction: str) -> list[bool]:
    if direction == "high":
        return [math.isfinite(value) and value >= threshold for value in values]
    return [math.isfinite(value) and value <= threshold for value in values]


def random_margin(
    *,
    selected_count: int,
    positives: list[bool],
    good_controls: list[bool],
    observed_ba: float,
    label: str,
    feature: str,
    trials: int = 256,
) -> tuple[float, float]:
    n = len(positives)
    if selected_count <= 0 or selected_count > n or not math.isfinite(observed_ba):
        return math.nan, math.nan
    bas: list[float] = []
    indices = list(range(n))
    rng = deterministic_rng(label, feature, str(selected_count))
    for _ in range(trials):
        chosen = set(rng.sample(indices, selected_count))
        pred = [idx in chosen for idx in indices]
        recall, fpr, _, _ = score_prediction(pred, positives, good_controls, [""] * n)
        bas.append(balanced_accuracy(recall, fpr))
    rand = mean(bas)
    return observed_ba - rand if math.isfinite(rand) else math.nan, rand


def score_prediction(
    predicted: list[bool],
    positives: list[bool],
    good_controls: list[bool],
    seqs: list[str],
) -> tuple[float, float, int, int]:
    pos_n = sum(positives)
    good_n = sum(good_controls)
    tp = sum(1 for p, y in zip(predicted, positives) if p and y)
    fp_good = sum(1 for p, g in zip(predicted, good_controls) if p and g)
    recall = tp / pos_n if pos_n else math.nan
    good_fpr = fp_good / good_n if good_n else math.nan
    seq_cov = len({seq for pred, y, seq in zip(predicted, positives, seqs) if pred and y and seq})
    return recall, good_fpr, seq_cov, sum(predicted)


def build_case_rows() -> list[dict[str, Any]]:
    base = by_case(read_rows(BASE_ROWS))
    rgb = by_case(read_rows(RGB_ROWS))
    traj = by_case(read_rows(TRAJ_ROWS))
    local = by_case(read_rows(LOCAL_ROWS))
    rows: list[dict[str, Any]] = []
    for case_id, row in sorted(base.items()):
        rr = rgb.get(case_id, {})
        tr = traj.get(case_id, {})
        lr = local.get(case_id, {})
        merged: dict[str, Any] = {
            **row,
            "stable_common_seed_count": f(rr.get("stable_common_seed_count")),
            "risk_frac": f(rr.get("curr_dynamic_boundary_lowconf_frac")),
            "trajectory_focus_error_m": f(tr.get("focus_aligned_error_m")),
            "trajectory_boundary_mean_error_m": f(tr.get("boundary_mean_error_m")),
            "trajectory_boundary_max_error_m": f(tr.get("boundary_max_error_m")),
            "local_point_residual_mean": f(lr.get("local_point_residual_mean")),
            "local_point_residual_p50": f(lr.get("local_point_residual_p50")),
            "local_point_residual_p90": f(lr.get("local_point_residual_p90")),
            "local_point_residual_max": f(lr.get("local_point_residual_max")),
            "valid_pixel_count": f(lr.get("valid_pixel_count")),
            "local_point_geometry_source_id": lr.get("geometry_source_id", ""),
            "strict_visual_panel": str(lr.get("strict_visual_panel", "")),
        }
        merged.update(labels_for(merged))
        rows.append(merged)
    return rows


def evaluate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    features = [
        ("risk_frac", "high", "reject/unreliable evidence proxy"),
        ("stable_common_seed_count", "low", "low Stage-C stable seed support"),
        ("local_point_residual_mean", "high", "high local point residual"),
        ("local_point_residual_p90", "high", "high local point residual p90"),
        ("local_point_residual_max", "high", "high local point residual max"),
        ("valid_pixel_count", "low", "low valid local geometry support"),
        ("trajectory_focus_error_m", "high", "high trajectory focus error"),
        ("trajectory_boundary_mean_error_m", "high", "high boundary trajectory error"),
        ("trajectory_boundary_max_error_m", "high", "high boundary max trajectory error"),
    ]
    target_labels = ["swa_handoff", "read_local", "non_good_target", "high_l3_non_good"]
    out: list[dict[str, Any]] = []
    seqs = [str(row.get("seq", "")) for row in rows]
    good_controls = [bool(row.get("safe_good_control")) for row in rows]
    for target in target_labels:
        positives = [bool(row.get(target)) for row in rows]
        for feature, direction, description in features:
            values = [f(row.get(feature)) for row in rows]
            thresholds = sorted({value for value in values if math.isfinite(value)})
            best: dict[str, Any] | None = None
            for threshold in thresholds:
                pred = select(values, threshold, direction)
                recall, good_fpr, seq_cov, selected_count = score_prediction(pred, positives, good_controls, seqs)
                ba = balanced_accuracy(recall, good_fpr)
                margin, rand_ba = random_margin(
                    selected_count=selected_count,
                    positives=positives,
                    good_controls=good_controls,
                    observed_ba=ba,
                    label=target,
                    feature=feature,
                )
                row = {
                    "target_label": target,
                    "feature": feature,
                    "feature_description": description,
                    "direction": direction,
                    "threshold": threshold,
                    "positive_count": sum(positives),
                    "good_control_count": sum(good_controls),
                    "selected_count": selected_count,
                    "bad_recall": recall,
                    "good_FPR": good_fpr,
                    "sequence_coverage": seq_cov,
                    "balanced_accuracy": ba,
                    "same_count_random_balanced_accuracy": rand_ba,
                    "same_count_random_margin": margin,
                    "selected_cases": ";".join(row["case_id"] for row, chosen in zip(rows, pred) if chosen),
                    "true_positive_cases": ";".join(row["case_id"] for row, chosen, pos in zip(rows, pred, positives) if chosen and pos),
                    "good_false_positive_cases": ";".join(row["case_id"] for row, chosen, good in zip(rows, pred, good_controls) if chosen and good),
                    "selected_case_oracle_pass": (
                        bool(sum(positives) >= 3)
                        and math.isfinite(recall)
                        and recall >= 0.65
                        and math.isfinite(good_fpr)
                        and good_fpr <= 0.25
                        and seq_cov >= 2
                        and math.isfinite(margin)
                        and margin >= 0.05
                    ),
                    "strict_promotion_allowed": False,
                    "strict_blocker": "selected-base-case diagnostic only; no full target-universe semantic rotation / anchor-id query-head control rerun",
                }
                if best is None:
                    best = row
                else:
                    key = (
                        f(row["selected_case_oracle_pass"]),
                        f(row["balanced_accuracy"]),
                        f(row["same_count_random_margin"]),
                        -abs(f(row["selected_count"]) - f(row["positive_count"])),
                    )
                    best_key = (
                        f(best["selected_case_oracle_pass"]),
                        f(best["balanced_accuracy"]),
                        f(best["same_count_random_margin"]),
                        -abs(f(best["selected_count"]) - f(best["positive_count"])),
                    )
                    if key > best_key:
                        best = row
            if best is not None:
                out.append(best)
    return sorted(out, key=lambda r: (not bool(r["selected_case_oracle_pass"]), -f(r["balanced_accuracy"]), str(r["target_label"]), str(r["feature"])))


def control_readiness(best: dict[str, Any]) -> list[dict[str, Any]]:
    selected = [item for item in str(best.get("selected_cases", "")).split(";") if item]
    true_positive = {item for item in str(best.get("true_positive_cases", "")).split(";") if item}
    v101_join = by_case(read_rows(V101_JOIN_ROWS))
    rows: list[dict[str, Any]] = []
    for case_id in selected:
        v101_run = V101_TARGET28_ROOT / case_id / "READ_NO_ACTION"
        v102_run = V102_REPAIR_TRACE_ROOT / case_id / "READ_NO_ACTION"
        trace_count = len(list((v101_run / "swa_raw_transport_trace").glob("*.pt"))) if (v101_run / "swa_raw_transport_trace").is_dir() else 0
        sidecar_count = len(list((v101_run / "per_chunk_geometry").glob("chunk_*.pt"))) if (v101_run / "per_chunk_geometry").is_dir() else 0
        repair_trace_count = len(list((v102_run / "swa_raw_transport_trace").glob("*.pt"))) if (v102_run / "swa_raw_transport_trace").is_dir() else 0
        repair_sidecar_count = len(list((v102_run / "per_chunk_geometry").glob("chunk_*.pt"))) if (v102_run / "per_chunk_geometry").is_dir() else 0
        join = v101_join.get(case_id, {})
        target_tax = str(join.get("target_taxonomy", ""))
        reasons = []
        if not (trace_count or repair_trace_count):
            reasons.append("missing_stage_c_trace")
        if sidecar_count < 2 and repair_sidecar_count < 2:
            reasons.append("missing_per_chunk_geometry_sidecars")
        if not join:
            reasons.append("missing_v101_lifecycle_geometry_join_row")
        elif target_tax != "HANDOFF_SCALE_GAUGE_TARGET":
            reasons.append(f"not_handoff_target_taxonomy:{target_tax}")
        if f(join.get("combined_geometry_unique_coverage"), 0.0) < 0.10:
            reasons.append("low_combined_geometry_unique_coverage")
        rows.append(
            {
                "case_id": case_id,
                "selected_by_best_candidate": True,
                "true_positive_under_best_candidate": case_id in true_positive,
                "v101_target28_trace_count": trace_count,
                "v101_target28_sidecar_count": sidecar_count,
                "v102_repair_trace_count": repair_trace_count,
                "v102_repair_sidecar_count": repair_sidecar_count,
                "has_trace_for_control_rerun": bool(trace_count or repair_trace_count),
                "has_sidecars_for_control_rerun": bool(sidecar_count >= 2 or repair_sidecar_count >= 2),
                "in_v101_lifecycle_geometry_join": bool(join),
                "v101_target_taxonomy": target_tax,
                "v101_combined_geometry_unique_coverage": join.get("combined_geometry_unique_coverage", ""),
                "v101_true_geometry_source_available_frac": join.get("true_geometry_source_available_frac", ""),
                "full_control_rerun_ready": not reasons,
                "blocking_reasons": ";".join(reasons),
            }
        )
    return rows


def main() -> int:
    case_rows = build_case_rows()
    eval_rows = evaluate(case_rows)
    write_rows(CASE_OUT, case_rows)
    write_rows(ROWS_OUT, eval_rows)
    passing = [row for row in eval_rows if row.get("selected_case_oracle_pass")]
    semantic_features = {"risk_frac", "stable_common_seed_count"}
    semantic_passing = [row for row in passing if row.get("feature") in semantic_features]
    geometry_passing = [row for row in passing if row.get("feature") not in semantic_features]
    best = eval_rows[0] if eval_rows else {}
    readiness_rows = control_readiness(best)
    write_rows(READINESS_OUT, readiness_rows)
    summary = {
        "schema": "acl2_v102_stage3_local_geometry_oracle_repair_v1",
        "case_count": len(case_rows),
        "strict_visual_case_count": sum(str(row.get("strict_visual_panel")).lower() == "true" for row in case_rows),
        "feature_candidate_count": len(eval_rows),
        "selected_case_oracle_pass_count": len(passing),
        "selected_case_oracle_pass_labels": sorted({row["target_label"] for row in passing}),
        "semantic_specific_selected_case_oracle_pass_count": len(semantic_passing),
        "geometry_only_selected_case_oracle_pass_count": len(geometry_passing),
        "best_candidate": best,
        "full_control_readiness_case_count": len(readiness_rows),
        "full_control_ready_case_count": sum(1 for row in readiness_rows if row.get("full_control_rerun_ready")),
        "full_control_readiness_blockers": sorted(
            {
                reason
                for row in readiness_rows
                for reason in str(row.get("blocking_reasons", "")).split(";")
                if reason
            }
        ),
        "strict_semantic_oracle_pass": False,
        "runtime_action_allowed": False,
        "strict_blocker": (
            "selected-base-case local geometry diagnostics do not replace full target-universe controls; "
            "semantic-specific risk/stable-seed features have no selected-case pass, and semantic rotation / anchor-id query-head control reruns are missing"
        ),
        "outputs": {
            "case_rows": CASE_OUT.as_posix(),
            "candidate_rows": ROWS_OUT.as_posix(),
            "full_control_readiness_rows": READINESS_OUT.as_posix(),
            "report": REPORT.as_posix(),
        },
    }
    write_json(SUMMARY, summary)
    report_lines = [
        "# Stage3 Local Geometry Oracle Repair",
        "",
        f"- case_count: {summary['case_count']}",
        f"- strict_visual_case_count: {summary['strict_visual_case_count']}",
        f"- selected_case_oracle_pass_count: {summary['selected_case_oracle_pass_count']}",
        f"- semantic_specific_selected_case_oracle_pass_count: {summary['semantic_specific_selected_case_oracle_pass_count']}",
        f"- geometry_only_selected_case_oracle_pass_count: {summary['geometry_only_selected_case_oracle_pass_count']}",
        f"- full_control_ready_case_count: {summary['full_control_ready_case_count']}/{summary['full_control_readiness_case_count']}",
        f"- strict_semantic_oracle_pass: {summary['strict_semantic_oracle_pass']}",
        f"- runtime_action_allowed: {summary['runtime_action_allowed']}",
        "",
        "This audit uses true local point residual maps and trajectory/RGB semantic panels for the selected v102 base cases only.",
        "It is useful fail-forward evidence, but it does not authorize Stage4/5/6 because full target-universe controls are not rerun here.",
        "The passing selected-case candidates are geometry-only trajectory/local-point features; semantic-specific risk/stable-seed features do not pass this audit.",
        "",
        "Best candidate:",
        "",
        "```json",
        json.dumps(jsonable(best), indent=2, sort_keys=True),
        "```",
    ]
    write_text(REPORT, "\n".join(report_lines))
    print(json.dumps(jsonable(summary), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
