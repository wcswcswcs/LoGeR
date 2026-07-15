#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from build_v113hs_action_metric_summary import finite_values, rel_improvement
from build_v113hs_baseline_metric_summary import summarize_sequence


ROOT = Path(__file__).resolve().parents[1]
V113_ROOT = ROOT / "results/acl2_v113hs_horizonstream_semantic_aware_geometric_evidence_influence"
RESULT_ROOT = ROOT / "results/acl2_v115tf_semantic_internal_alignment_evidence_influence_control"
DIAG = RESULT_ROOT / "diagnostics"
STAGE1 = RESULT_ROOT / "stage1_hook_audit"
REPORTS = RESULT_ROOT / "reports"

GQ1_CHUNKBLOCK_PREFIX = "stage_hs_gq1_layer23_state_delta_gain_chunkblock1_rerun1_notrace"


def rel(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def summarize_pair(seq: str) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    baseline_root = V113_ROOT / "outputs" / f"baseline_kitti_{seq}"
    candidate_root = RESULT_ROOT / "outputs" / f"{GQ1_CHUNKBLOCK_PREFIX}_full_kitti_{seq}"
    base = summarize_sequence(baseline_root, seq)
    cand = summarize_sequence(candidate_root, seq)
    base["variant"] = "v113_baseline_default_no_loop"
    cand["variant"] = "HS_GQ1_layer23_state_delta_gain_semantic_internal_mild_chunkblock1_notrace"

    cmp_row: dict[str, Any] = {
        "seq": seq,
        "baseline_variant": base["variant"],
        "candidate_variant": cand["variant"],
        "baseline_output_root": rel(baseline_root),
        "candidate_output_root": rel(candidate_root),
    }
    for metric in [
        "full_ATE_sim3_rmse",
        "rolling_ate_p90",
        "final_error_sim3_aligned",
        "segment_scale_log_error_median_abs",
        "adjacent_log_scale_jump_p90_abs",
        "rpe_delta1_translation_mean",
        "rpe_delta1_rotation_deg_mean",
        "global_sim3_scale",
    ]:
        b = base.get(metric)
        c = cand.get(metric)
        cmp_row[f"baseline_{metric}"] = b
        cmp_row[f"candidate_{metric}"] = c
        cmp_row[f"{metric}_rel_improvement"] = rel_improvement(b, c)
        if b is not None and c is not None:
            cmp_row[f"{metric}_abs_delta_candidate_minus_baseline"] = float(c - b)
    return base, cand, cmp_row


def gq1_chunkblock_metric_summary(seqs: list[str]) -> dict[str, Any]:
    metric_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []
    for seq in seqs:
        base, cand, cmp_row = summarize_pair(seq)
        metric_rows.extend([base, cand])
        comparison_rows.append(cmp_row)

    full_ate_improvements = finite_values(comparison_rows, "full_ATE_sim3_rmse_rel_improvement")
    rolling_improvements = finite_values(comparison_rows, "rolling_ate_p90_rel_improvement")
    segment_improvements = finite_values(comparison_rows, "segment_scale_log_error_median_abs_rel_improvement")
    full_ate_harms = [max(0.0, -v) for v in full_ate_improvements]
    median_improvement_threshold = 0.05
    max_full_ate_harm_rel_threshold = 0.01

    aggregate = {
        "schema": "acl2_v115tf_hs_gq1_chunkblock1_metric_summary_v1",
        "seqs": seqs,
        "baseline_name": "v113_baseline_default_no_loop",
        "candidate_name": "HS_GQ1_layer23_state_delta_gain_semantic_internal_mild_chunkblock1_notrace",
        "claim_boundary": (
            "Reduced-memory config-specific pilot: chunk_block_num=1, gq_layer_filter=23, "
            "trace/audit disabled. This is not a default HorizonStream full-sequence success."
        ),
        "config_specific": True,
        "default_full_status": "OOM_BLOCKED",
        "median_full_ATE_rel_improvement": float(np.median(full_ate_improvements)) if full_ate_improvements else None,
        "median_rolling_p90_rel_improvement": float(np.median(rolling_improvements)) if rolling_improvements else None,
        "median_segment_scale_rel_improvement": float(np.median(segment_improvements)) if segment_improvements else None,
        "max_full_ATE_harm_rel": float(max(full_ate_harms)) if full_ate_harms else None,
        "improved_seq_count_full_ATE": int(sum(v > 0 for v in full_ate_improvements)),
        "segment_scale_not_worse_all": bool(segment_improvements and all(v >= 0 for v in segment_improvements)),
        "pilot_geometry_gate": {
            "thresholds": {
                "median_full_ATE_rel_improvement_ge": median_improvement_threshold,
                "or_median_rolling_p90_rel_improvement_ge": median_improvement_threshold,
                "max_full_ATE_harm_rel_le": max_full_ate_harm_rel_threshold,
                "segment_scale_not_worse_all": True,
            },
            "pass": bool(
                full_ate_improvements
                and rolling_improvements
                and segment_improvements
                and (
                    float(np.median(full_ate_improvements)) >= median_improvement_threshold
                    or float(np.median(rolling_improvements)) >= median_improvement_threshold
                )
                and max(full_ate_harms) <= max_full_ate_harm_rel_threshold
                and all(v >= 0 for v in segment_improvements)
            ),
        },
        "semantic_causality_gate": {
            "status": "not_run",
            "pass": False,
            "reason": "Geometry pilot gate did not pass, so semantic controls are not triggered.",
        },
    }
    write_csv(DIAG / f"{GQ1_CHUNKBLOCK_PREFIX}_metrics_rows.csv", metric_rows)
    write_csv(DIAG / f"{GQ1_CHUNKBLOCK_PREFIX}_comparison_rows.csv", comparison_rows)
    summary = {"metric_rows": metric_rows, "comparison_rows": comparison_rows, "aggregate": aggregate}
    write_json(DIAG / f"{GQ1_CHUNKBLOCK_PREFIX}_summary.json", summary)
    return summary


OOM_RE = re.compile(
    r"CUDA out of memory\. Tried to allocate\s+([0-9.]+\s+MiB).*?of which\s+([0-9.]+\s+MiB)\s+is free",
    re.S,
)


def log_oom_summary(log_path: Path) -> dict[str, Any]:
    if not log_path.exists():
        return {"oom": False, "oom_requested_mib": "", "oom_free_mib": "", "oom_line": ""}
    text = log_path.read_text(encoding="utf-8", errors="replace")
    match = OOM_RE.search(text)
    if not match:
        return {"oom": False, "oom_requested_mib": "", "oom_free_mib": "", "oom_line": ""}
    line = next((ln.strip() for ln in text.splitlines() if "CUDA out of memory" in ln), "")
    return {
        "oom": True,
        "oom_requested_mib": match.group(1).strip(),
        "oom_free_mib": match.group(2).strip(),
        "oom_line": line[:500],
    }


def manifest_status(case: str, branch: str, seq: str, repair: str) -> dict[str, Any]:
    manifest_path = DIAG / case / "run_manifest.json"
    data = read_json(manifest_path)
    log_path = Path(data.get("log_path", "")) if data.get("log_path") else RESULT_ROOT / "logs" / f"{case}.log"
    status = {
        "branch": branch,
        "seq": seq,
        "repair": repair,
        "case": case,
        "returncode": data.get("returncode", ""),
        "action": data.get("action", ""),
        "gpu": data.get("gpu", ""),
        "trace_enable": data.get("trace_enable", ""),
        "trace_gla_enable": data.get("trace_gla_enable", ""),
        "action_audit_enable": data.get("action_audit_enable", ""),
        "gq_layer_filter": data.get("gq_layer_filter", ""),
        "chunk_block_num": data.get("chunk_block_num", ""),
        "config_path": rel(Path(data["config_path"])) if data.get("config_path") else "",
        "manifest": rel(manifest_path),
        "log_path": rel(log_path),
        "output_root": rel(Path(data["output_root"])) if data.get("output_root") else "",
    }
    status.update(log_oom_summary(log_path))
    return status


def gq_full_status_rows() -> list[dict[str, Any]]:
    specs = [
        ("GQ3", "stage_hs_gq3_pre_gla_pose_token_gain_risk_semantic_internal_medium_rerun1_full_kitti_{seq}", "default trace/audit full"),
        ("GQ3", "stage_hs_gq3_pre_gla_pose_token_gain_risk_semantic_internal_medium_rerun2_inplace_full_kitti_{seq}", "in-place token scaling full"),
        ("GQ4", "stage_hs_gq4_pre_gla_mrt_token_gain_risk_semantic_internal_medium_rerun1_full_kitti_{seq}", "MRT-token target trace/audit full"),
        ("GQ4", "stage_hs_gq4_pre_gla_mrt_token_gain_risk_semantic_internal_medium_rerun2_notrace_full_kitti_{seq}", "MRT-token target no-trace/no-audit full"),
        ("GQ4", "stage_hs_gq4_layer23_pre_gla_mrt_token_gain_risk_semantic_internal_medium_rerun1_full_kitti_{seq}", "layer23 trace/audit full"),
        ("GQ4", "stage_hs_gq4_layer23_pre_gla_mrt_token_gain_risk_semantic_internal_medium_rerun2_notrace_full_kitti_{seq}", "layer23 no-trace/no-audit full"),
        ("GQ1", "stage_hs_gq1_state_delta_gain_semantic_internal_mild_rerun1_full_kitti_{seq}", "state-delta all-layer trace/audit full"),
        ("GQ1", "stage_hs_gq1_state_delta_gain_semantic_internal_mild_rerun2_notrace_full_kitti_{seq}", "state-delta all-layer no-trace/no-audit full"),
        ("GQ1", "stage_hs_gq1_layer23_state_delta_gain_semantic_internal_mild_rerun1_notrace_full_kitti_{seq}", "state-delta layer23 no-trace/no-audit full"),
        ("GQ1", "stage_hs_gq1_layer23_state_delta_gain_semantic_internal_mild_rerun2_inplace_notrace_full_kitti_{seq}", "state-delta layer23 in-place no-trace/no-audit full"),
        ("GQ1", f"{GQ1_CHUNKBLOCK_PREFIX}_full_kitti_{{seq}}", "state-delta layer23 chunk_block_num=1 no-trace/no-audit full"),
    ]
    rows: list[dict[str, Any]] = []
    for branch, template, repair in specs:
        for seq in ["00", "02"]:
            rows.append(manifest_status(template.format(seq=seq), branch, seq, repair))
    return rows


def numeric_stats_from_csv(path: Path, fields: list[str]) -> dict[str, Any]:
    if not path.exists():
        return {"path": rel(path), "row_count": 0}
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    out: dict[str, Any] = {"path": rel(path), "row_count": len(rows)}
    for field in fields:
        vals = []
        for row in rows:
            raw = row.get(field, "")
            if raw == "":
                continue
            try:
                vals.append(float(raw))
            except ValueError:
                continue
        if vals:
            out[f"{field}_min"] = float(np.min(vals))
            out[f"{field}_median"] = float(np.median(vals))
            out[f"{field}_max"] = float(np.max(vals))
    layers = sorted({int(float(row["global_layer_idx"])) for row in rows if row.get("global_layer_idx") not in ("", None)})
    out["global_layer_idx_values"] = layers
    changed = [row.get("changed_state") for row in rows if row.get("changed_state") != ""]
    if changed:
        out["changed_state_true_count"] = sum(str(v).lower() == "true" for v in changed)
    return out


def smoke_summaries() -> dict[str, Any]:
    return {
        "gq3_pre_gla_pose_token_smoke": read_json(STAGE1 / "hs_gq3_smoke_gate_summary.json"),
        "gq4_pre_gla_mrt_token_smoke": read_json(STAGE1 / "hs_gq4_smoke_gate_summary.json"),
        "gq4_layer23_pre_gla_mrt_token_smoke": read_json(STAGE1 / "hs_gq4_layer23_smoke_gate_summary.json"),
        "gq1_state_delta_all_layer_smoke": numeric_stats_from_csv(
            DIAG / "smoke_hs_gq1_state_delta_gain_semantic_internal_mild_max12_kitti_00/hs_gq_state_action_rows.csv",
            ["state_delta_gate", "state_delta_rel_norm_raw", "state_delta_rel_norm_after", "semantic_risk_mean", "semantic_stable_mean"],
        ),
        "gq1_state_delta_layer23_smoke": numeric_stats_from_csv(
            DIAG / "smoke_hs_gq1_layer23_state_delta_gain_semantic_internal_mild_max12_kitti_00/hs_gq_state_action_rows.csv",
            ["state_delta_gate", "state_delta_rel_norm_raw", "state_delta_rel_norm_after", "semantic_risk_mean", "semantic_stable_mean"],
        ),
        "gq1_state_delta_layer23_inplace_smoke": numeric_stats_from_csv(
            DIAG / "smoke_hs_gq1_layer23_state_delta_gain_semantic_internal_mild_inplace_max12_kitti_00/hs_gq_state_action_rows.csv",
            ["state_delta_gate", "state_delta_rel_norm_raw", "state_delta_rel_norm_after", "semantic_risk_mean", "semantic_stable_mean"],
        ),
        "gq1_state_delta_layer23_chunkblock1_smoke": numeric_stats_from_csv(
            DIAG / "smoke_hs_gq1_layer23_state_delta_gain_chunkblock1_max12_kitti_00/hs_gq_state_action_rows.csv",
            ["state_delta_gate", "state_delta_rel_norm_raw", "state_delta_rel_norm_after", "semantic_risk_mean", "semantic_stable_mean"],
        ),
    }


def fmt_pct(value: Any) -> str:
    if value is None or value == "":
        return "NA"
    return f"{float(value) * 100:.3f}%"


def write_report(metric_summary: dict[str, Any], status_rows: list[dict[str, Any]]) -> None:
    aggregate = metric_summary["aggregate"]
    comparison_rows = metric_summary["comparison_rows"]
    lines = [
        "# ACL2 v115TF HS-GQ Decision Summary",
        "",
        "Boundary: GQ1 chunk_block_num=1 is a reduced-memory config-specific pilot, not a default HorizonStream full success.",
        "",
        "## GQ1 Chunkblock1 Metric Comparison",
        "",
        "| seq | baseline full ATE | candidate full ATE | full ATE rel | rolling p90 rel | segment scale rel |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in comparison_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["seq"]),
                    f"{float(row['baseline_full_ATE_sim3_rmse']):.6f}",
                    f"{float(row['candidate_full_ATE_sim3_rmse']):.6f}",
                    fmt_pct(row["full_ATE_sim3_rmse_rel_improvement"]),
                    fmt_pct(row["rolling_ate_p90_rel_improvement"]),
                    fmt_pct(row["segment_scale_log_error_median_abs_rel_improvement"]),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            f"Pilot geometry gate pass: `{aggregate['pilot_geometry_gate']['pass']}`.",
            f"Median full ATE relative improvement: `{fmt_pct(aggregate['median_full_ATE_rel_improvement'])}`.",
            f"Median rolling p90 relative improvement: `{fmt_pct(aggregate['median_rolling_p90_rel_improvement'])}`.",
            f"Segment scale not worse all: `{aggregate['segment_scale_not_worse_all']}`.",
            "",
            "## Full/OOM Status",
            "",
            "| branch | repair | seq | returncode | requested | free | chunk | layer |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in status_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["branch"]),
                    str(row["repair"]),
                    str(row["seq"]),
                    str(row["returncode"]),
                    str(row["oom_requested_mib"]),
                    str(row["oom_free_mib"]),
                    str(row["chunk_block_num"]),
                    str(row["gq_layer_filter"]),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "Conclusion: GQ state/token actions are real at smoke/action level, but default full promotion remains OOM-blocked. "
            "The only full 00/02 GQ1 metric run uses chunk_block_num=1 and still misses the 5% geometry gate, so controls are not triggered.",
            "",
        ]
    )
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "V115_HS_GQ_DECISION_SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    metric_summary = gq1_chunkblock_metric_summary(["00", "02"])
    status_rows = gq_full_status_rows()
    write_csv(DIAG / "stage_hs_gq_full_status_rows.csv", status_rows)
    summary = {
        "schema": "acl2_v115tf_hs_gq_decision_summary_v1",
        "gq1_chunkblock_metric_summary": metric_summary["aggregate"],
        "gq_smoke_summaries": smoke_summaries(),
        "gq_full_status_rows": status_rows,
        "decision": {
            "default_full_status": "OOM_BLOCKED",
            "config_specific_full_pilot": True,
            "controls_required": False,
            "controls_required_reason": "No GQ candidate passed the geometry pilot gate under an allowed/default-success boundary.",
            "final_gq_classification": "GQ_CONFIG_SPECIFIC_NO_GEOMETRY_EFFECT_AND_DEFAULT_OOM_BLOCKED",
        },
        "outputs": {
            "metric_rows": rel(DIAG / f"{GQ1_CHUNKBLOCK_PREFIX}_metrics_rows.csv"),
            "comparison_rows": rel(DIAG / f"{GQ1_CHUNKBLOCK_PREFIX}_comparison_rows.csv"),
            "metric_summary": rel(DIAG / f"{GQ1_CHUNKBLOCK_PREFIX}_summary.json"),
            "full_status_rows": rel(DIAG / "stage_hs_gq_full_status_rows.csv"),
            "report": rel(REPORTS / "V115_HS_GQ_DECISION_SUMMARY.md"),
        },
    }
    write_json(DIAG / "stage_hs_gq_decision_summary.json", summary)
    write_report(metric_summary, status_rows)
    print(json.dumps(summary["decision"] | metric_summary["aggregate"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
