#!/usr/bin/env python3
"""Build ACL2 v116-TF Stage0 evidence-freeze artifacts.

This is a read-only freeze of prior v110R/v111/v112/v113/v114/v115 evidence.
It records what can be reused, what is only a historical boundary, and which
controls/hooks must be repaired or rerun in v116.
"""

from __future__ import annotations

import csv
import json
import math
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "results/acl2_v116tf_fast_semantic_causal_memory_influence"
OUT = RESULT_ROOT / "stage0_evidence_freeze"

V110 = ROOT / "results/acl2_v110r_lingbot_multisurface_full_ate_boost_semantic_causality"
V111 = ROOT / "results/acl2_v111tf_lingbot_semantic_aware_memory_management_anchor_local_trajectory"
V112 = ROOT / "results/acl2_v112tf_lingbot_semantic_aware_memory_management_expansion_horizon_augmented"
V113 = ROOT / "results/acl2_v113hs_horizonstream_semantic_aware_geometric_evidence_influence"
V114 = ROOT / "results/acl2_v114tf_semantic_internal_evidence_quality_memory_influence_control"
V115 = ROOT / "results/acl2_v115tf_semantic_internal_alignment_evidence_influence_control"

SEQS_4 = ("00", "01", "02", "05")
SEQS_PILOT = ("00", "02")


def rel(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def clean_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clean_json(v) for v in value]
    if isinstance(value, Path):
        return rel(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(clean_json(data), indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def fnum(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def boolish(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "pass"}


def median(values: list[float]) -> float:
    vals = sorted(v for v in values if math.isfinite(v))
    if not vals:
        return float("nan")
    mid = len(vals) // 2
    if len(vals) % 2:
        return vals[mid]
    return 0.5 * (vals[mid - 1] + vals[mid])


def mean(values: list[float]) -> float:
    vals = [v for v in values if math.isfinite(v)]
    return sum(vals) / len(vals) if vals else float("nan")


def max_harm(values: list[float]) -> float:
    vals = [v for v in values if math.isfinite(v)]
    return max([max(0.0, -v) for v in vals], default=float("nan"))


def artifact_row(artifact_id: str, path: Path, required: bool = True) -> dict[str, Any]:
    return {
        "artifact_id": artifact_id,
        "path": rel(path),
        "required": required,
        "exists": path.exists(),
        "size_bytes": path.stat().st_size if path.exists() else "",
    }


def artifact_manifest() -> list[dict[str, Any]]:
    artifacts = [
        ("v110_final_decision", V110 / "final_decision/final_decision.json", True),
        ("v110_b1_stage4_full_metrics", V110 / "stage4_full_00_01_02_05_validation/full_metric_rows.csv", True),
        ("v110_b1_stage4_action_fidelity", V110 / "stage4_full_00_01_02_05_validation/action_fidelity_rows.csv", True),
        ("v110_b1_stage8_control_configs", V110 / "stage8_b1_full_controls/stage8_config_generation_summary.json", True),
        ("v110_b1_stage8_full_metrics", V110 / "stage8_b1_full_controls/full_metric_rows.csv", False),
        ("v111_a1_metric_summary", V111 / "batch_a_a1_anchor_selection/a1_metric_summary.json", True),
        ("v111_a1_full_metrics", V111 / "batch_a_a1_anchor_selection/full_metric_rows.csv", True),
        ("v112_stage0_summary", V112 / "stage0_evidence_freeze/stage0_summary.json", False),
        ("v113_hs_value_decision", V113 / "diagnostics/stage6_action_decision_summary.json", True),
        ("v113_hs_semantic_projection_seq00", V113 / "semantic_projection/seq00_risk.npy", True),
        ("v113_hs_semantic_projection_seq02", V113 / "semantic_projection/seq02_risk.npy", True),
        ("v114_hs_lq_decision", V114 / "diagnostics/stage_hs_lq_decision_summary.json", True),
        ("v114_hs_lq_rows", V114 / "diagnostics/stage_hs_lq_decision_rows.csv", True),
        ("v115_hook_audit_summary", V115 / "stage1_hook_audit/stage1_hook_audit_summary.json", True),
        ("v115_hs_la_blocker", V115 / "stage1_hook_audit/source_span_audit_reports/HS_LOCAL_ATTENTION_LOGIT_HOOK_BLOCKED.md", True),
        ("v115_l2_query_full_summary", V115 / "stage5_lingbot_a2_l2_query_full_pilot_00_02/query_full_metric_summary.json", True),
        ("v115_l2_control_summary", V115 / "stage5_lingbot_l2_special_weight_repair_00_02/l2_control_metric_summary.json", True),
        ("v115_hs_final_decision_rows", V115 / "diagnostics/stage_hs_final_decision_rows.csv", True),
        ("v115_hs_gq_decision", V115 / "diagnostics/stage_hs_gq_decision_summary.json", True),
        ("lingbot_wrapper", ROOT / "third_party/lingbot-map/benchmark/methods/lingbot_map.py", True),
        ("lingbot_attention", ROOT / "third_party/lingbot-map/lingbot_map/layers/attention.py", True),
        ("horizonstream_runtime", ROOT / "third_party/HorizonStream/horizonstream/runtime/semantic_runtime.py", True),
        ("horizonstream_model", ROOT / "third_party/HorizonStream/horizonstream/models/horizonstream.py", True),
        ("horizonstream_runner", ROOT / "tools/run_v115tf_hs_deterministic_pipeline.py", True),
    ]
    return [artifact_row(*item) for item in artifacts]


def summarize_metric_rows(rows: list[dict[str, str]], policy_id: str, seqs: tuple[str, ...] = SEQS_4) -> dict[str, Any]:
    selected = [r for r in rows if r.get("policy_id") == policy_id and r.get("seq") in seqs]
    rels = [fnum(r.get("full_ATE_sim3_relative_improvement_vs_baseline")) for r in selected]
    rolling = [fnum(r.get("rolling_ATE_p90_relative_improvement_vs_baseline")) for r in selected]
    local = [fnum(r.get("local_window_ATE_rel_improvement_vs_baseline_median")) for r in selected]
    return {
        "policy_id": policy_id,
        "seqs": ",".join(seqs),
        "row_count": len(selected),
        "seq_full_rel": {r.get("seq", ""): fnum(r.get("full_ATE_sim3_relative_improvement_vs_baseline")) for r in selected},
        "median_full_rel": median(rels),
        "mean_full_rel": mean(rels),
        "improved_seq_count": sum(1 for v in rels if math.isfinite(v) and v > 0.0),
        "max_harm": max_harm(rels),
        "median_rolling_p90_rel": median(rolling),
        "median_local_window_rel": median(local),
    }


def baseline_summary_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    v110_full = read_csv(V110 / "stage4_full_00_01_02_05_validation/full_metric_rows.csv")
    b1 = summarize_metric_rows(v110_full, "B1_semantic_only", SEQS_4)
    rows.append(
        {
            "branch": "Task1_AB_reference",
            "model": "LingBot",
            "source_version": "v110R",
            "candidate_or_fact": "B1_semantic_only",
            "median_full_rel": b1["median_full_rel"],
            "mean_full_rel": b1["mean_full_rel"],
            "improved_seq_count": b1["improved_seq_count"],
            "max_harm": b1["max_harm"],
            "semantic_status": "strong_geometry_internal_schedule_baseline_only",
            "source_path": rel(V110 / "stage4_full_00_01_02_05_validation/full_metric_rows.csv"),
            "v116_use": "reference geometry lever for A1+B1 comparison; not semantic success",
        }
    )

    stage8_summary = read_json(V110 / "stage8_b1_full_controls/stage8_config_generation_summary.json")
    rows.append(
        {
            "branch": "Task1_AB_reference_controls",
            "model": "LingBot",
            "source_version": "v110R",
            "candidate_or_fact": "B1_stage8_control_configs",
            "median_full_rel": "",
            "mean_full_rel": "",
            "improved_seq_count": "",
            "max_harm": "",
            "semantic_status": "configs_ready_metrics_not_collated_in_stage8",
            "source_path": rel(V110 / "stage8_b1_full_controls/stage8_config_generation_summary.json"),
            "v116_use": f"control families={','.join(stage8_summary.get('policy_families', []))}; rerun/collate if needed",
        }
    )

    a1_summary = read_json(V111 / "batch_a_a1_anchor_selection/a1_metric_summary.json")
    rows.append(
        {
            "branch": "Task1_AB_reference",
            "model": "LingBot",
            "source_version": "v111TF",
            "candidate_or_fact": a1_summary.get("best_semantic_policy_by_median_vs_default", ""),
            "median_full_rel": a1_summary.get("best_semantic_policy_median_full_rel_vs_a1_default", ""),
            "mean_full_rel": a1_summary.get("best_semantic_policy_mean_full_rel_vs_a1_default", ""),
            "improved_seq_count": "",
            "max_harm": "",
            "semantic_status": "semantic_random_p95_pass" if a1_summary.get("semantic_causality_claim_allowed") else "check_summary",
            "source_path": rel(V111 / "batch_a_a1_anchor_selection/a1_metric_summary.json"),
            "v116_use": "cleaner anchor reference for AB composition",
        }
    )

    l2_query = read_json(V115 / "stage5_lingbot_a2_l2_query_full_pilot_00_02/query_full_metric_summary.json")
    rows.append(
        {
            "branch": "Task2_L2T_reference",
            "model": "LingBot",
            "source_version": "v115TF",
            "candidate_or_fact": "LB_L2_special_query_local_read_family",
            "median_full_rel": "",
            "mean_full_rel": "",
            "improved_seq_count": "",
            "max_harm": "",
            "semantic_status": l2_query.get("taxonomy", ""),
            "source_path": rel(V115 / "stage5_lingbot_a2_l2_query_full_pilot_00_02/query_full_metric_summary.json"),
            "v116_use": "L2 clean/weak reference; must upgrade to token-level source and selected-query logits",
        }
    )

    l2_control = read_json(V115 / "stage5_lingbot_l2_special_weight_repair_00_02/l2_control_metric_summary.json")
    rows.append(
        {
            "branch": "Task2_L2T_reference",
            "model": "LingBot",
            "source_version": "v115TF",
            "candidate_or_fact": l2_control.get("candidate_policy_id", ""),
            "median_full_rel": l2_control.get("candidate_median_full_rel", ""),
            "mean_full_rel": "",
            "improved_seq_count": "",
            "max_harm": "",
            "semantic_status": l2_control.get("taxonomy", ""),
            "source_path": rel(V115 / "stage5_lingbot_l2_special_weight_repair_00_02/l2_control_metric_summary.json"),
            "v116_use": "negative repair boundary; do not reuse frame/source-level special weight as L2T",
        }
    )

    hook = read_json(V115 / "stage1_hook_audit/stage1_hook_audit_summary.json")
    rows.append(
        {
            "branch": "Task3_HS_LA_reference",
            "model": "HorizonStream",
            "source_version": "v115TF",
            "candidate_or_fact": "HS_LA_attention_logit_hook",
            "median_full_rel": "",
            "mean_full_rel": "",
            "improved_seq_count": "",
            "max_harm": "",
            "semantic_status": hook.get("hs_la_attention_logit_status", ""),
            "source_path": rel(V115 / "stage1_hook_audit/stage1_hook_audit_summary.json"),
            "v116_use": "must repair selected-query recompute; HS-HG is not a substitute",
        }
    )

    for row in read_csv(V115 / "diagnostics/stage_hs_final_decision_rows.csv"):
        rows.append(
            {
                "branch": "Task3_HS_LA_reference",
                "model": "HorizonStream",
                "source_version": "v115TF",
                "candidate_or_fact": row.get("candidate_name", row.get("label", "")),
                "median_full_rel": row.get("median_full_ATE_rel_improvement", ""),
                "mean_full_rel": "",
                "improved_seq_count": "",
                "max_harm": row.get("max_full_ATE_harm_rel", ""),
                "semantic_status": "v115_cleaner_but_weak_or_reduced_config",
                "source_path": rel(V115 / "diagnostics/stage_hs_final_decision_rows.csv"),
                "v116_use": "historical non-value weak surface; cannot claim v116 success",
            }
        )

    v113_decision = read_json(V113 / "diagnostics/stage6_action_decision_summary.json")
    for row in v113_decision.get("rows", []):
        rows.append(
            {
                "branch": "Task4_HS_SC_reference",
                "model": "HorizonStream",
                "source_version": "v113HS",
                "candidate_or_fact": row.get("name", ""),
                "median_full_rel": row.get("median_full_ATE_rel_improvement", ""),
                "mean_full_rel": "",
                "improved_seq_count": row.get("improved_seq_count_full_ATE", ""),
                "max_harm": row.get("max_full_ATE_harm_rel", ""),
                "semantic_status": row.get("decision", ""),
                "source_path": rel(V113 / "diagnostics/stage6_action_decision_summary.json"),
                "v116_use": "value path strong but controls matched; sign contrast needed",
            }
        )

    v114_decision = read_json(V114 / "diagnostics/stage_hs_lq_decision_summary.json")
    rows.append(
        {
            "branch": "Task4_HS_SC_reference",
            "model": "HorizonStream",
            "source_version": "v114TF",
            "candidate_or_fact": "HS_LQ_generic_rowmean_confounded_boundary",
            "median_full_rel": "",
            "mean_full_rel": "",
            "improved_seq_count": "",
            "max_harm": "",
            "semantic_status": v114_decision.get("final_taxonomy", ""),
            "source_path": rel(V114 / "diagnostics/stage_hs_lq_decision_summary.json"),
            "v116_use": "same-magnitude semantic sign contrast is required before value-path semantic claim",
        }
    )
    return rows


def stale_process_rows() -> list[str]:
    proc = subprocess.run(
        ["ps", "-eo", "pid,ppid,stat,etime,cmd"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    markers = (
        "third_party/HorizonStream",
        "third_party/lingbot-map",
        "horizonstream",
        "lingbot",
        "v110r",
        "v111tf",
        "v112tf",
        "v113hs",
        "v114tf",
        "v115tf",
        "v116tf",
    )
    self_markers = (
        "build_v116tf_stage0_evidence_freeze.py",
        "ps -eo pid,ppid,stat,etime,cmd",
        "rg -i",
        "grep",
    )
    out: list[str] = []
    for line in proc.stdout.splitlines():
        if not any(marker in line for marker in markers):
            continue
        if any(marker in line for marker in self_markers):
            continue
        out.append(line.strip())
    return out


def blocker_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    hook_summary = read_json(V115 / "stage1_hook_audit/stage1_hook_audit_summary.json")
    rows.append(
        {
            "branch": "Task3_HS_LA",
            "failure_type": "HOOK_BLOCKED",
            "primary_blocker": "v115 default path uses fused SDPA and did not expose full attention/logit map",
            "source_path": rel(V115 / "stage1_hook_audit/source_span_audit_reports/HS_LOCAL_ATTENTION_LOGIT_HOOK_BLOCKED.md"),
            "status": hook_summary.get("hs_la_attention_logit_status", ""),
            "v116_required_attempt": "implement selected-query QK/V recompute and replace selected query outputs",
            "disallowed_attempt": "return to HS-HG head-output gate as substitute",
        }
    )
    rows.append(
        {
            "branch": "Task2_L2T",
            "failure_type": "ACTION_SCOPE_UPGRADE_REQUIRED",
            "primary_blocker": "v115 L2 was frame/source-level special weighting and weak/non-positive after repair",
            "source_path": rel(V115 / "stage5_lingbot_l2_special_weight_repair_00_02/l2_control_metric_summary.json"),
            "status": read_json(V115 / "stage5_lingbot_l2_special_weight_repair_00_02/l2_control_metric_summary.json").get("taxonomy", ""),
            "v116_required_attempt": "prove local_window_context token-level source spans and special-query-only target mapping",
            "disallowed_attempt": "run guessed frame-level value scaling and label it L2T",
        }
    )
    rows.append(
        {
            "branch": "Task4_HS_SC",
            "failure_type": "CONTROL_MATCHED_HISTORY",
            "primary_blocker": "v113/v114 value-path geometry gains were matched or exceeded by generic controls",
            "source_path": rel(V113 / "diagnostics/stage6_action_decision_summary.json"),
            "status": "same_magnitude_sign_contrast_required",
            "v116_required_attempt": "run correct/reverse/random/rowmean same-magnitude contrast",
            "disallowed_attempt": "claim value-path semantic causality from correct-sign geometry alone",
        }
    )
    return rows


def write_report(summary: dict[str, Any], rows: list[dict[str, Any]], blockers: list[dict[str, Any]]) -> None:
    lines = [
        "# ACL2 v116-TF Stage0 Evidence Freeze",
        "",
        f"- stage0_complete: `{summary['stage0_complete']}`",
        f"- missing_required_count: `{summary['missing_required_count']}`",
        f"- stale_process_count: `{summary['stale_process_count']}`",
        "",
        "## Baseline/Boundary Facts",
        "",
        "| branch | model | source | candidate_or_fact | median_full_rel | semantic_status | v116_use |",
        "|---|---|---|---:|---:|---|---|",
    ]
    for row in rows:
        lines.append(
            "| {branch} | {model} | {source_version} | {candidate_or_fact} | {median_full_rel} | {semantic_status} | {v116_use} |".format(
                **{k: str(row.get(k, "")).replace("|", "/") for k in row}
            )
        )
    lines += [
        "",
        "## Blockers And Required v116 Attempts",
        "",
        "| branch | failure_type | primary_blocker | v116_required_attempt | disallowed_attempt |",
        "|---|---|---|---|---|",
    ]
    for row in blockers:
        lines.append(
            "| {branch} | {failure_type} | {primary_blocker} | {v116_required_attempt} | {disallowed_attempt} |".format(
                **{k: str(row.get(k, "")).replace("|", "/") for k in row}
            )
        )
    lines += [
        "",
        "## Audit Note",
        "",
        "Stage0 is evidence freeze only. It does not create new geometry results and does not resolve any blocker.",
    ]
    write_text(OUT / "STAGE0_EVIDENCE_FREEZE_REPORT.md", "\n".join(lines))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    artifacts = artifact_manifest()
    required_missing = [row for row in artifacts if row["required"] and not row["exists"]]
    optional_missing = [row for row in artifacts if not row["required"] and not row["exists"]]
    rows = baseline_summary_rows()
    blockers = blocker_rows()
    stale = stale_process_rows()

    summary = {
        "schema": "acl2_v116tf_stage0_evidence_freeze_summary_v1",
        "result_root": rel(RESULT_ROOT),
        "stage0_complete": not required_missing,
        "missing_required_count": len(required_missing),
        "missing_optional_count": len(optional_missing),
        "stale_process_count": len(stale),
        "required_missing": required_missing,
        "optional_missing": optional_missing,
        "stale_process_rows": stale,
        "key_boundaries": {
            "b1_reference": "strong geometry, semantic causality not established",
            "a1_reference": "cleaner semantic anchor positive reference",
            "l2_reference": "v115 clean/weak or negative repair; must become token-level in v116",
            "hs_la_reference": "v115 attention/logit blocked; selected-query recompute required",
            "hs_value_reference": "v113/v114 generic controls confounded value path; sign contrast required",
        },
    }
    write_json(OUT / "STAGE0_EVIDENCE_FREEZE_SUMMARY.json", summary)
    write_csv(OUT / "STAGE0_ARTIFACT_MANIFEST.csv", artifacts)
    write_csv(OUT / "STAGE0_BASELINE_BOUNDARY_ROWS.csv", rows)
    write_csv(OUT / "STAGE0_BLOCKER_ROWS.csv", blockers)
    write_report(summary, rows, blockers)
    print(json.dumps(clean_json(summary), indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
