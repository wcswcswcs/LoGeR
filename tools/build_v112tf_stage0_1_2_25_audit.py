#!/usr/bin/env python3
"""Build ACL2 v112TF Stage0/1/2/2.5 audit and diagnostic artifacts.

This builder is intentionally read-only with respect to previous experiments.
It freezes v109-v111 references, audits LingBot memory-management hooks, builds
memory-specific semantic cue rows, and creates proxy H0/H3 diagnostics. Runtime
action experiments are generated/executed by later branch-specific tools.
"""

from __future__ import annotations

import csv
import json
import math
import os
import statistics
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "results/acl2_v112tf_lingbot_semantic_aware_memory_management_expansion_horizon_augmented"
STAGE0 = RESULT_ROOT / "stage0_evidence_freeze"
STAGE1 = RESULT_ROOT / "stage1_hook_traceability_audit"
STAGE2 = RESULT_ROOT / "stage2_memory_specific_cue_bank"
STAGE25 = RESULT_ROOT / "stage25_h0_h3_diagnostics"

V105 = ROOT / "results/acl2_v105tf_dualtrack_lingbot_loger_evidence_eligibility"
V108 = ROOT / "results/acl2_v108tf_lingbot_memory_operation_action_surface_search"
V109 = ROOT / "results/acl2_v109tf_lingbot_f_surface_causal_dissection_semantic_memory_control"
V110 = ROOT / "results/acl2_v110r_lingbot_multisurface_full_ate_boost_semantic_causality"
V111 = ROOT / "results/acl2_v111tf_lingbot_semantic_aware_memory_management_anchor_local_trajectory"

V108_STAGE2 = V108 / "stage2_semantic_cue_bank"
V109_F19 = V109 / "stage2_role_specific_safety_candidates"
V109_F19_CONTROLS = V109 / "stage2_f19_keyframe_controls"
V110_STAGE4 = V110 / "stage4_full_00_01_02_05_validation"
V110_FINAL = V110 / "final_decision"
V111_STAGE0 = V111 / "stage0_evidence_freeze"
V111_STAGE1 = V111 / "stage1_alignment_and_hook_audit"
V111_T1 = V111 / "batch_t_t1_b1_core_controls"
V111_T2 = V111 / "batch_t_t2_context_token_ablation"
V111_T3 = V111 / "batch_t_t3_soft_token_weighting"
V111_A1 = V111 / "batch_a_a1_anchor_selection"

SEQUENCES = ("00", "01", "02", "05")
EXPECTED_BASELINE = {
    "00": 46.00057328153847,
    "01": 57.097417656974685,
    "02": 77.48077587398916,
    "05": 19.9612565679075,
}
B1_POLICY = "B1_semantic_only"
F19_POLICY = "F19_dynamic_or_special_admitted_high_risk_else_weak_context"
A1_POLICY = "A1_low_dynamic_from_first32"


def read_csv(path: Path, *, limit: int | None = None) -> list[dict[str, str]]:
    if not path.exists():
        return []
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(row)
            if limit is not None and len(rows) >= limit:
                break
    return rows


def iter_csv(path: Path):
    if not path.exists():
        return
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            yield row


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


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


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(clean_json(data), indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


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


def rel(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def fnum(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def boolish(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "pass"}


def mean(values: list[float]) -> float:
    vals = [v for v in values if math.isfinite(v)]
    return sum(vals) / len(vals) if vals else float("nan")


def median(values: list[float]) -> float:
    vals = sorted(v for v in values if math.isfinite(v))
    if not vals:
        return float("nan")
    mid = len(vals) // 2
    return vals[mid] if len(vals) % 2 else 0.5 * (vals[mid - 1] + vals[mid])


def max_harm(values: list[float]) -> float:
    vals = [v for v in values if math.isfinite(v)]
    return max([max(0.0, -v) for v in vals], default=float("nan"))


def source_locus(path: Path, needle: str) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
    line_no = ""
    for idx, line in enumerate(text.splitlines(), start=1):
        if needle in line:
            line_no = idx
            break
    return {
        "path": rel(path),
        "needle": needle,
        "exists": path.exists(),
        "needle_found": bool(line_no),
        "line": line_no,
    }


def latest_phase_counts(run_results: Path) -> dict[str, Any]:
    rows = read_csv(run_results)
    latest: dict[tuple[str, str], dict[str, str]] = {}
    historical: dict[str, dict[str, int]] = {}
    for row in rows:
        phase = row.get("phase", "")
        run_name = row.get("run_name", "")
        rc = row.get("returncode", "")
        if not phase or not run_name:
            continue
        latest[(phase, run_name)] = row
        historical.setdefault(phase, {}).setdefault(str(rc), 0)
        historical[phase][str(rc)] += 1
    by_phase: dict[str, dict[str, int]] = {}
    for (phase, _run_name), row in latest.items():
        rc = row.get("returncode", "")
        by_phase.setdefault(phase, {}).setdefault(str(rc), 0)
        by_phase[phase][str(rc)] += 1
    return {"latest": by_phase, "historical": historical, "row_count": len(rows)}


def stale_process_rows() -> list[str]:
    proc = subprocess.run(
        ["ps", "-eo", "pid,ppid,stat,etime,cmd"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    markers = (
        "third_party/lingbot-map/benchmark",
        "run_v108tf_gpu_serial_policy_manifest.py",
        "run_v111tf",
        "ACL2_V108_STAGE4_POLICY_ID",
        "ACL2_V112",
    )
    self_markers = (
        "build_v112tf_stage0_1_2_25_audit.py",
        "ps -eo pid,ppid,stat,etime,cmd",
    )
    out: list[str] = []
    for line in proc.stdout.splitlines():
        if not any(marker in line for marker in markers):
            continue
        if any(marker in line for marker in self_markers):
            continue
        out.append(line.strip())
    return out


def metric_aggregate(path: Path, policy_col: str, policy_id: str, rel_cols: list[str]) -> dict[str, Any]:
    rows = [row for row in read_csv(path) if row.get(policy_col) == policy_id]
    rels: list[float] = []
    for row in rows:
        for col in rel_cols:
            value = fnum(row.get(col))
            if math.isfinite(value):
                rels.append(value)
                break
    return {
        "policy_id": policy_id,
        "source": rel(path),
        "row_count": len(rows),
        "median_full_rel": median(rels),
        "mean_full_rel": mean(rels),
        "improved_seq_count": sum(1 for value in rels if math.isfinite(value) and value > 0.0),
        "max_harm": max_harm(rels),
        "per_seq_rel": {
            row.get("seq", row.get("seq_id", "")): next(
                (fnum(row.get(col)) for col in rel_cols if math.isfinite(fnum(row.get(col)))),
                float("nan"),
            )
            for row in rows
        },
    }


def build_stage0() -> dict[str, Any]:
    STAGE0.mkdir(parents=True, exist_ok=True)
    artifact_rows = []
    artifacts = [
        ("v105_baseline", V105 / "stage1_lingbot_baseline/full_sequence_metrics/lingbot_stream_default_full_metrics.csv", "required"),
        ("v108_frame_semantic_summary", V108_STAGE2 / "frame_semantic_summary.csv", "required"),
        ("v108_token_semantic_rows", V108_STAGE2 / "token_semantic_rows.csv", "required"),
        ("v108_operation_semantic_summary", V108_STAGE2 / "operation_semantic_summary.csv", "required"),
        ("v109_f19_full_metric_rows", V109_F19 / "full_metric_rows.csv", "required"),
        ("v109_f19_keyframe_control_summary", V109_F19_CONTROLS / "f19_keyframe_control_summary.json", "required"),
        ("v110_b1_full_metric_rows", V110_STAGE4 / "full_metric_rows.csv", "required"),
        ("v110_final_decision", V110_FINAL / "final_decision.json", "required"),
        ("v111_stage0_summary", V111_STAGE0 / "stage0_summary.json", "required"),
        ("v111_stage1_summary", V111_STAGE1 / "stage1_summary.json", "required"),
        ("v111_t1_summary", V111_T1 / "t1_core_summary.json", "required"),
        ("v111_t2_summary", V111_T2 / "t2_metric_summary.json", "required"),
        ("v111_t3_summary", V111_T3 / "t3_metric_summary.json", "required"),
        ("v111_a1_summary", V111_A1 / "a1_metric_summary.json", "required"),
        ("v111_a1_full_metric_rows", V111_A1 / "full_metric_rows.csv", "required"),
        ("lingbot_wrapper", ROOT / "third_party/lingbot-map/benchmark/methods/lingbot_map.py", "required"),
        ("lingbot_stream_model", ROOT / "third_party/lingbot-map/lingbot_map/models/gct_stream.py", "required"),
        ("lingbot_attention_layer", ROOT / "third_party/lingbot-map/lingbot_map/layers/attention.py", "required"),
    ]
    for artifact_id, path, requirement in artifacts:
        size = path.stat().st_size if path.exists() else ""
        row_count: int | str = ""
        if path.exists() and path.suffix == ".csv" and isinstance(size, int) and size < 100_000_000:
            row_count = len(read_csv(path))
        elif path.exists() and path.suffix == ".csv":
            row_count = "skipped_large_csv"
        artifact_rows.append(
            {
                "schema": "acl2_v112tf_stage0_artifact_manifest_row_v1",
                "artifact_id": artifact_id,
                "path": rel(path),
                "requirement": requirement,
                "exists": path.exists(),
                "size_bytes": size,
                "row_count": row_count,
            }
        )

    baseline_rows = []
    baseline_exact: dict[str, Any] = {}
    for row in read_csv(artifacts[0][1]):
        seq = row.get("seq", "")
        if seq not in EXPECTED_BASELINE:
            continue
        observed = fnum(row.get("ATE_full_sim3_m"))
        expected = EXPECTED_BASELINE[seq]
        abs_diff = abs(observed - expected)
        baseline_exact[seq] = {
            "observed": observed,
            "expected": expected,
            "abs_diff": abs_diff,
            "match": math.isfinite(abs_diff) and abs_diff <= 1e-12,
        }
        baseline_rows.append(
            {
                "schema": "acl2_v112tf_stage0_baseline_row_v1",
                "seq": seq,
                "ATE_full_sim3_m": observed,
                "expected_ATE_full_sim3_m": expected,
                "abs_diff": abs_diff,
                "baseline_exact_match": math.isfinite(abs_diff) and abs_diff <= 1e-12,
                "source": rel(artifacts[0][1]),
            }
        )

    b1 = metric_aggregate(
        V110_STAGE4 / "full_metric_rows.csv",
        "policy_id",
        B1_POLICY,
        ["full_ATE_sim3_relative_improvement_vs_baseline"],
    )
    f19 = metric_aggregate(
        V109_F19 / "full_metric_rows.csv",
        "policy_id",
        F19_POLICY,
        ["full_ATE_sim3_relative_improvement_vs_baseline"],
    )
    a1 = metric_aggregate(
        V111_A1 / "full_metric_rows.csv",
        "policy_id",
        A1_POLICY,
        ["full_ATE_sim3_relative_improvement_vs_baseline"],
    )
    a1_summary = read_json(V111_A1 / "a1_metric_summary.json")
    t1_summary = read_json(V111_T1 / "t1_core_summary.json")
    t2_summary = read_json(V111_T2 / "t2_metric_summary.json")
    t3_summary = read_json(V111_T3 / "t3_metric_summary.json")
    semantic_summary = read_json(V108_STAGE2 / "stage2_summary.json")
    f19_control = read_json(V109_F19_CONTROLS / "f19_keyframe_control_summary.json")
    v110_final = read_json(V110_FINAL / "final_decision.json")
    pending = stale_process_rows()
    missing = [row["artifact_id"] for row in artifact_rows if row["requirement"] == "required" and not row["exists"]]

    references = [
        {"reference_id": "v110_B1", **b1, "semantic_boundary": "not semantic-aware; v110 final taxonomy internal/schedule baseline only"},
        {"reference_id": "v109_F19", **f19, "semantic_boundary": "keyframe controls match on multiple sequences; not semantic-aware"},
        {
            "reference_id": "v111_A1",
            **a1,
            "semantic_boundary": "A1_low_dynamic_from_first32 beats random same-first32 P95 but is delayed anchor initialization, not B1 replacement",
            "random_p95": a1_summary.get("random_same_first32_p95_median_full_rel_vs_a1_default"),
            "latency_frames_max": a1_summary.get("best_policy_latency_frames_max", ""),
        },
        {
            "reference_id": "v111_T1",
            "policy_id": "B1 core controls",
            "median_full_rel": t1_summary.get("best_policy_median_full_rel", t1_summary.get("semantic_plus_internal_median")),
            "semantic_boundary": t1_summary.get("blocker", "semantic_shuffle matched B1 core subset"),
            "source": rel(V111_T1 / "t1_core_summary.json"),
        },
        {
            "reference_id": "v111_T2",
            "policy_id": t2_summary.get("best_policy_by_median_full_rel", ""),
            "median_full_rel": t2_summary.get("best_policy_median_full_rel"),
            "taxonomy": t2_summary.get("taxonomy"),
            "semantic_boundary": t2_summary.get("semantic_causality_claim_blocker"),
            "source": rel(V111_T2 / "t2_metric_summary.json"),
        },
        {
            "reference_id": "v111_T3",
            "policy_id": t3_summary.get("best_policy_by_median_full_rel", ""),
            "median_full_rel": t3_summary.get("best_policy_median_full_rel"),
            "taxonomy": t3_summary.get("taxonomy"),
            "semantic_boundary": t3_summary.get("semantic_causality_claim_blocker"),
            "source": rel(V111_T3 / "t3_metric_summary.json"),
        },
    ]

    write_csv(STAGE0 / "available_artifact_manifest.csv", artifact_rows)
    write_csv(STAGE0 / "frozen_baseline_table.csv", baseline_rows)
    write_csv(STAGE0 / "reference_metric_rows.csv", references)
    write_text(
        STAGE0 / "forbidden_repeat_list.md",
        """# ACL2 v112TF Forbidden Repeat List

1. Do not claim v112 success by inheriting B1 or A1.
2. Do not merge geometry pass and semantic causality pass.
3. Do not run full KITTI while stale LingBot workers exist.
4. Do not use selected windows, debug96, trace movement, action fidelity, or proxy rows as full KITTI success.
5. Do not use global source tokens as Anchor/Local source spans when the hook cannot separate context.
6. Do not claim HorizonStream absorption unless at least two of H0/H2/H3 are completed or explicitly blocked with evidence.
""",
    )
    write_text(
        STAGE0 / "allowed_new_surfaces.md",
        """# ACL2 v112TF Allowed New Surfaces

- A1 variants: existing `anchor_scale_frame_indices` frame-selection hook.
- T4/H1 coarse semantic lifetime: existing `trajectory_context_token_mask` / per-frame `stage4_context_token_mask_map` can support compact context-token gates after parity.
- B1/T1 controls: existing `force_non_keyframe` hook, but semantic causality cannot be claimed without stronger controls.
- A2/A3/L1/L2/H2/T5/C1/D1 require hook audit or blocker before runtime claims.
""",
    )

    stage0_pass = (
        set(baseline_exact) == set(SEQUENCES)
        and all(item["match"] for item in baseline_exact.values())
        and math.isfinite(fnum(b1.get("median_full_rel")))
        and math.isfinite(fnum(a1_summary.get("best_policy_median_full_rel_vs_a1_default")))
        and bool(semantic_summary)
        and not pending
        and not missing
        and v110_final.get("final_taxonomy") == "FULL_ATE_BOOST_INTERNAL_SCHEDULE_BASELINE_ONLY"
        and f19_control.get("f19_keyframe_control_supports_f19_causality") is False
    )
    blockers: list[str] = []
    if pending:
        blockers.append("stale_lingbot_worker_running")
        write_text(
            STAGE0 / "STALE_WORKER_BLOCKER.md",
            "# STALE_WORKER_BLOCKER\n\n"
            "触发条件：Stage0 发现 LingBot/manifest worker 进程仍在运行。\n\n"
            f"pending_lingbot_process_rows:\n\n```text\n{os.linesep.join(pending)}\n```\n\n"
            "当前不能继续的原因：v112 full KITTI 运行可能与旧 worker 写同一 workspace 或占用 GPU。\n\n"
            "不能 claim：不能 claim Stage0 pass，也不能启动新 full KITTI。\n",
        )
    if missing:
        blockers.extend(f"missing_required_artifact:{item}" for item in missing)
        write_text(
            STAGE0 / "REFERENCE_ARTIFACT_MISSING.md",
            "# REFERENCE_ARTIFACT_MISSING\n\n"
            f"missing_required_artifacts: `{missing}`\n\n"
            "不能用复盘文字补数值，必须回到 result root 找 artifact 或重跑对应 builder。\n",
        )

    summary = {
        "schema": "acl2_v112tf_stage0_evidence_freeze_summary_v1",
        "stage0_pass": stage0_pass,
        "blockers": blockers,
        "baseline_exact_match": all(item["match"] for item in baseline_exact.values()) if baseline_exact else False,
        "baseline_exact_match_by_seq": baseline_exact,
        "b1_reference": b1,
        "f19_reference": f19,
        "a1_reference": {
            **a1,
            "best_policy_median_full_rel_vs_a1_default": a1_summary.get("best_policy_median_full_rel_vs_a1_default"),
            "random_same_first32_p95": a1_summary.get("random_same_first32_p95_median_full_rel_vs_a1_default"),
            "semantic_random_p95_pass_policy_ids": a1_summary.get("a1_semantic_random_p95_pass_policy_ids"),
        },
        "t1_boundary": t1_summary,
        "t2_boundary": t2_summary,
        "t3_boundary": t3_summary,
        "semantic_cue_bank_readable": bool(semantic_summary),
        "no_stale_lingbot_worker": not pending,
        "pending_lingbot_process_rows": pending,
        "required_missing_artifacts": missing,
        "outputs": {
            "artifact_manifest": rel(STAGE0 / "available_artifact_manifest.csv"),
            "frozen_baseline_table": rel(STAGE0 / "frozen_baseline_table.csv"),
            "reference_metric_rows": rel(STAGE0 / "reference_metric_rows.csv"),
            "forbidden_repeat_list": rel(STAGE0 / "forbidden_repeat_list.md"),
            "allowed_new_surfaces": rel(STAGE0 / "allowed_new_surfaces.md"),
            "summary": rel(STAGE0 / "stage0_summary.json"),
        },
    }
    write_json(STAGE0 / "stage0_summary.json", summary)
    return summary


def build_stage1() -> dict[str, Any]:
    STAGE1.mkdir(parents=True, exist_ok=True)
    wrapper = ROOT / "third_party/lingbot-map/benchmark/methods/lingbot_map.py"
    stream = ROOT / "third_party/lingbot-map/lingbot_map/models/gct_stream.py"
    attention = ROOT / "third_party/lingbot-map/lingbot_map/layers/attention.py"

    loci = [
        ("A1_anchor_frame_selection", "anchor_initialization", "existing_runtime_hook", wrapper, "stage4_scale_frame_indices", "default-off parity inherited from v111 A1_default_first_n vs v105"),
        ("A2_anchor_attention_bias", "anchor_read_attention", "blocked_no_action_hook", attention, "F.scaled_dot_product_attention(", "source role trace exists, but no configurable source-token attention bias hook"),
        ("A3_anchor_value_scaling", "anchor_value_scaling", "blocked_no_action_hook", attention, "v_full", "no configurable anchor value scaling hook; shape-safe minimal hook still needed"),
        ("L1_local_attention_bias", "local_read_attention", "blocked_no_action_hook", attention, "_v105_key_role", "trace can classify local_window_context, but no local source-token bias hook"),
        ("L2_query_type_specific_local_read", "local_read_attention", "blocked_query_type_hook", attention, "query_token_role", "trace samples query role, but no query-type-specific action dispatch"),
        ("L3_local_value_scaling", "local_value_scaling", "blocked_no_action_hook", attention, "v_full", "no local value scaling hook"),
        ("T4_context_token_mask", "trajectory_context_token_mask", "existing_runtime_hook", wrapper, "stage4_context_token_mask_map", "v111 T2/T3 parity proves all-ones and legacy equivalence for compact context token masks"),
        ("T5_trajectory_retrieval", "retrieval_selection", "blocked_no_retrieval_hook", attention, "compute_attention", "no query-key similarity based retrieval selection hook exposed"),
        ("C1_retention_eviction_ordering", "retention_ordering", "blocked_trace_only", attention, "_apply_kv_cache_eviction", "eviction trace exists, but no semantic retention/eviction ordering control"),
        ("D1_trajectory_write_admission", "trajectory_write_gate", "blocked_trace_only", attention, "trajectory_write", "trajectory_write trace exists, but no semantic write gate hook"),
        ("H0_influence_trace", "influence_diagnosis", "partial_trace_only", attention, "_v105_trace_sdpa_attention", "top-k sampled attention trace exists; full context-separated mass is not recorded"),
        ("H2_head_output_gate", "head_wise_reliability_gate", "blocked_no_head_output_hook", attention, "x = F.scaled_dot_product_attention", "head output y_h is not exposed before merge for semantic gate"),
        ("H3_scale_carrier_probe", "offline_probe", "proxy_only_available", attention, "_v107_emit_cache_append_rows", "hidden token feature dumps absent; operation/read-mass/norm proxy can be built"),
    ]
    code_rows = []
    for hook_id, operation_type, support_status, path, needle, note in loci:
        loc = source_locus(path, needle)
        code_rows.append(
            {
                "schema": "acl2_v112tf_stage1_code_locus_row_v1",
                "hook_id": hook_id,
                "operation_type": operation_type,
                "support_status": support_status,
                "path": loc["path"],
                "line": loc["line"],
                "needle": needle,
                "needle_found": loc["needle_found"],
                "note": note,
            }
        )

    hook_status = [
        {
            "schema": "acl2_v112tf_stage1_hook_status_row_v1",
            "hook": row["hook_id"],
            "context": row["operation_type"],
            "default_off_parity": {
                "A1_anchor_frame_selection": "pass_inherited_v111_A1_default_v105",
                "T4_context_token_mask": "pass_inherited_v111_T2_T3_all1_parity",
            }.get(row["hook_id"], "not_available"),
            "action_fidelity": {
                "A1_anchor_frame_selection": "pass_inherited_v111_A1_action_rows_for_promoted_policy",
                "T4_context_token_mask": "pass_inherited_v111_T2_T3_action_rows",
            }.get(row["hook_id"], "not_available"),
            "full_pilot_run": "not_run_in_v112",
            "blocker": "" if row["support_status"] in {"existing_runtime_hook", "proxy_only_available"} else row["support_status"],
        }
        for row in code_rows
    ]
    parity_rows = [
        {
            "schema": "acl2_v112tf_stage1_noop_parity_row_v1",
            "hook": "A1_anchor_frame_selection",
            "source": rel(V111_A1 / "a1_default_v105_parity_rows.csv"),
            "parity_scope": "default_first_n matches v105 baseline",
            "status": "pass_inherited",
        },
        {
            "schema": "acl2_v112tf_stage1_noop_parity_row_v1",
            "hook": "T4_context_token_mask",
            "source": rel(V111_T2 / "t2_parity_crosscheck_rows.csv"),
            "parity_scope": "default-off all1 and legacy equivalence for compact context-token masks",
            "status": "pass_inherited",
        },
        {
            "schema": "acl2_v112tf_stage1_noop_parity_row_v1",
            "hook": "A2/A3/L1/L2/L3/H2/T5/C1/D1",
            "source": rel(STAGE1 / "code_loci.csv"),
            "parity_scope": "new hook missing or trace-only",
            "status": "blocked_before_parity",
        },
    ]
    traceability_rows = [
        {
            "schema": "acl2_v112tf_stage1_traceability_row_v1",
            "trace_id": "v105_sdpa_topk_attention",
            "available": True,
            "context_fields": "key_context_role,key_frame_offset,key_token_role,query_token_role,attention_weight",
            "limitation": "top-k sampled heads/queries only; not full mass by context/head/source frame",
            "source": rel(attention),
        },
        {
            "schema": "acl2_v112tf_stage1_traceability_row_v1",
            "trace_id": "v107_cache_operation",
            "available": True,
            "context_fields": "operation_type,context_path,token_type,source_frame,source_age,retention_region",
            "limitation": "operation/read proxy only; no attention head output y_h",
            "source": rel(attention),
        },
    ]

    write_csv(STAGE1 / "code_loci.csv", code_rows)
    write_csv(STAGE1 / "hook_status_rows.csv", hook_status)
    write_csv(STAGE1 / "noop_parity_manifest.csv", parity_rows)
    write_csv(STAGE1 / "traceability_rows.csv", traceability_rows)
    write_text(
        STAGE1 / "hook_contract.md",
        """# ACL2 v112TF Hook Contract

Existing runtime hooks usable after inherited parity:
- A1 `anchor_scale_frame_indices`
- T4/H1 compact context-token masks through `stage4_context_token_mask` and `stage4_context_token_mask_map`
- B1/T1 `force_non_keyframe`

Blocked before runtime claim:
- A2/A3 anchor attention/value hooks
- L1/L2/L3 local attention/value/query-type hooks
- H2 head-wise output gate
- T5 retrieval selection
- C1/D1 retention/write control
""",
    )

    blocker_specs = [
        (
            "A2_ANCHOR_SOURCE_SPAN_BLOCKED.md",
            "A2/A3 cannot run as attention/value action",
            "Source classification is trace-only through `_v105_key_role`; no config-dispatched anchor-context source-token bias/value scaling hook exists.",
            "Implement shape-preserving source-token bias/value hook in SDPA path; expose anchor/local/trajectory source spans; run default-off parity on 00/02.",
        ),
        (
            "QUERY_TYPE_INDEX_BLOCKED.md",
            "L2/H2 cannot run as query-type-specific action",
            "Query role is only sampled in top-k trace; the forward path has no query-type-specific action dispatch or full query index mask.",
            "Expose query role mask for camera/register/anchor/patch; implement identity default-off path; verify local_window_ATE parity.",
        ),
        (
            "H0_INFLUENCE_TRACE_BLOCKED.md",
            "H0 full influence mass is blocked",
            "Existing trace records top-k sampled attention, not total context-separated attention mass, source-age mass, or per-head dynamic/stable mass.",
            "Add full mass aggregation inside `_v105_trace_sdpa_attention`; keep row count bounded by aggregating per context/head/query type.",
        ),
        (
            "H2_HEAD_OUTPUT_HOOK_BLOCKED.md",
            "H2 head-wise gate cannot run",
            "Head output y_h is merged before a semantic reliability gate can be applied; no hook exposes g_h or changed head count.",
            "Insert gate between SDPA output and head merge; add default-off all-one parity; record head_count_changed and mean_g_h.",
        ),
        (
            "T5_RETRIEVAL_HOOK_BLOCKED.md",
            "T5 retrieval cannot run",
            "No query-key similarity based trajectory-memory retrieval selection hook is exposed; cache append/eviction is schedule based.",
            "Audit k_full source-frame map; expose top-k trajectory source selection; begin with proxy-only read_frequency/source_age retrieval.",
        ),
        (
            "C1_D1_HOOK_BLOCKED_REPORT.md",
            "C1/D1 are trace-only",
            "Retention/eviction/trajectory_write rows exist, but there is no semantic keep/drop/write gate that changes ordering or admission.",
            "Add semantic retention-order and write-admission maps; verify default-off parity; compare C1/D1 against B1/T4.",
        ),
    ]
    for filename, trigger, checked, repair in blocker_specs:
        write_text(
            STAGE1 / filename,
            f"""# {filename.removesuffix('.md')}

触发条件：{trigger}

已检查文件 / function / config：
- `{rel(wrapper)}`
- `{rel(stream)}`
- `{rel(attention)}`
- `code_loci.csv`

失败的具体字段或指标：
{checked}

当前不能继续的原因：
没有通过 default-off parity 的 runtime hook，继续跑 full ATE 会把未定义 action 当成方法结果。

可继续尝试的修复方向：
1. {repair}
2. 先跑 seq00/02 no-action parity smoke，再进入 00/02 pilot。
3. 产出 action_fidelity_rows，记录 changed token/head/source span。

哪些结论不能 claim：
不能 claim 该 branch 科学失败；只能 claim 当前是 engineering hook/source-span blocker。
""",
        )

    blockers = [
        row["hook"] for row in hook_status if str(row["blocker"]).startswith("blocked") or row["blocker"] == "partial_trace_only"
    ]
    summary = {
        "schema": "acl2_v112tf_stage1_hook_traceability_audit_summary_v1",
        "stage1_pass": True,
        "stage1_full_action_ready": False,
        "a1_hook_ready": True,
        "t4_h1_context_token_mask_ready": True,
        "h0_traceability_status": "partial_trace_only",
        "h3_probe_status": "proxy_only_available",
        "blocked_or_trace_only_hooks": blockers,
        "non_b1_a1_new_surface_ready_for_00_02_pilot": "T4/H1 compact context token mask can be configured after Stage2 cue rows",
        "outputs": {
            "code_loci": rel(STAGE1 / "code_loci.csv"),
            "hook_status_rows": rel(STAGE1 / "hook_status_rows.csv"),
            "noop_parity_manifest": rel(STAGE1 / "noop_parity_manifest.csv"),
            "traceability_rows": rel(STAGE1 / "traceability_rows.csv"),
            "hook_contract": rel(STAGE1 / "hook_contract.md"),
        },
    }
    write_json(STAGE1 / "stage1_summary.json", summary)
    return summary


def role_value(row: dict[str, str], name: str) -> float:
    return fnum(row.get(name), 0.0)


def anchor_q(row: dict[str, str]) -> float:
    return (
        1.0 * role_value(row, "stable_structure_mass")
        - 1.5 * role_value(row, "dynamic_mass")
        - 1.0 * role_value(row, "boundary_mass")
        - 1.0 * role_value(row, "raw_unknown_lowtrust_trust_mass")
        - 0.3 * role_value(row, "weak_context_mass")
    )


def traj_q(row: dict[str, str]) -> float:
    return (
        1.0 * role_value(row, "stable_structure_mass")
        + 0.5 * role_value(row, "semantic_continuity_score")
        - 1.5 * role_value(row, "dynamic_mass")
        - 1.0 * role_value(row, "boundary_mass")
        - 0.8 * role_value(row, "raw_unknown_lowtrust_trust_mass")
        - 0.4 * role_value(row, "weak_context_mass")
    )


def sigmoid(x: float) -> float:
    if x < -60:
        return 0.0
    if x > 60:
        return 1.0
    return 1.0 / (1.0 + math.exp(-x))


def lifetime_gates(row: dict[str, str]) -> dict[str, float]:
    risk = role_value(row, "dynamic_mass") + role_value(row, "boundary_mass") + 0.7 * role_value(row, "raw_unknown_lowtrust_trust_mass") + 0.3 * role_value(row, "weak_context_mass")
    support = role_value(row, "stable_structure_mass") + 0.5 * role_value(row, "semantic_continuity_score")
    base = support - risk
    return {
        "semantic_lifetime_risk": risk,
        "semantic_lifetime_support": support,
        "g_camera": max(0.0, min(1.0, sigmoid(2.0 * base))),
        "g_register": max(0.0, min(1.0, sigmoid(1.5 * base))),
        "g_anchor": max(0.0, min(1.0, sigmoid(1.0 * base + 0.5))),
    }


def build_stage2() -> dict[str, Any]:
    STAGE2.mkdir(parents=True, exist_ok=True)
    frame_path = V108_STAGE2 / "frame_semantic_summary.csv"
    op_path = V108_STAGE2 / "operation_semantic_summary.csv"
    token_path = V108_STAGE2 / "token_semantic_rows.csv"

    anchor_rows: list[dict[str, Any]] = []
    frame_scores: dict[tuple[str, int], dict[str, Any]] = {}
    seq_counts: dict[str, int] = {}
    for row in iter_csv(frame_path) or []:
        seq = row.get("seq_id", "")
        if seq not in SEQUENCES:
            continue
        frame = int(float(row.get("frame_id", 0)))
        gates = lifetime_gates(row)
        out = {
            "schema": "acl2_v112tf_anchor_memory_cue_row_v1",
            "seq": seq,
            "frame_id": frame,
            "latency_frames": frame + 1,
            "is_default_anchor_frame": frame < 8,
            "is_first16_candidate": frame < 16,
            "is_first24_candidate": frame < 24,
            "is_first32_candidate": frame < 32,
            "is_first64_candidate": frame < 64,
            "stable_landmark_mass": row.get("stable_structure_mass", ""),
            "dynamic_mass": row.get("dynamic_mass", ""),
            "boundary_mass": row.get("boundary_mass", ""),
            "weak_context_mass": row.get("weak_context_mass", ""),
            "road_ground_mass": row.get("road_ground_mass", ""),
            "sky_lowobs_mass": row.get("sky_lowobs_mass", ""),
            "unknown_lowtrust_mass": row.get("raw_unknown_lowtrust_trust_mass", ""),
            "semantic_trust_mean": row.get("semantic_trust_mean", ""),
            "semantic_purity_mean": row.get("semantic_purity_mean", ""),
            "semantic_continuity_score": row.get("semantic_continuity_score", ""),
            "Q_anchor_frame": anchor_q(row),
            "Q_traj_frame": traj_q(row),
            **gates,
        }
        anchor_rows.append(out)
        frame_scores[(seq, frame)] = out
        seq_counts[seq] = seq_counts.get(seq, 0) + 1

    local_rows: list[dict[str, Any]] = []
    traj_rows: list[dict[str, Any]] = []
    op_rows_total = 0
    for row in iter_csv(op_path) or []:
        seq = row.get("seq_id", "")
        if seq not in SEQUENCES:
            continue
        op_rows_total += 1
        current_frame = row.get("current_frame", "")
        source_frame = row.get("source_frame", "")
        context_path = row.get("context_path", "")
        token_type = row.get("token_type", "")
        base = {
            "seq": seq,
            "operation_type": row.get("operation_type", ""),
            "plan_operation_type": row.get("plan_operation_type", ""),
            "context_type": context_path,
            "token_type": token_type,
            "token_id": row.get("token_id", ""),
            "token_count": row.get("token_count", ""),
            "current_frame": current_frame,
            "source_frame": source_frame,
            "source_frame_age": row.get("source_frame_age", ""),
            "stable_landmark_mass": row.get("stable_structure_mass_mean", ""),
            "dynamic_mass": row.get("dynamic_mass_mean", ""),
            "boundary_mass": row.get("boundary_mass_mean", ""),
            "weak_context_mass": row.get("weak_context_mass_mean", ""),
            "road_ground_mass": row.get("road_ground_mass_mean", ""),
            "semantic_trust_mean": row.get("semantic_trust_mean_mean", ""),
            "semantic_purity_mean": row.get("semantic_purity_mean_mean", ""),
            "semantic_boundary_risk": row.get("semantic_boundary_risk_mean", ""),
            "semantic_continuity_score": row.get("semantic_continuity_score_mean", ""),
            "row_granularity": "operation_token_group_not_individual_patch_token",
        }
        if context_path == "local_pose_reference_window":
            risk = role_value(base, "dynamic_mass") + role_value(base, "boundary_mass") + 0.7 * role_value(base, "semantic_boundary_risk") + 0.2 * role_value(base, "weak_context_mass")
            support = 0.8 * role_value(base, "stable_landmark_mass")
            local_rows.append(
                {
                    "schema": "acl2_v112tf_local_window_token_cue_row_v1",
                    **base,
                    "local_dynamic_risk": role_value(base, "dynamic_mass"),
                    "local_boundary_risk": role_value(base, "boundary_mass"),
                    "local_lowtrust_risk": role_value(base, "semantic_boundary_risk"),
                    "local_stable_support": support,
                    "R_local": risk,
                    "S_local": support,
                    "w_special_query_mild": max(0.25, min(1.25, math.exp(-1.0 * risk + 0.25 * support))),
                    "w_all_query_mild": max(0.25, min(1.25, math.exp(-0.7 * risk + 0.2 * support))),
                    "query_type": "unknown_until_query_specific_hook",
                }
            )
        if context_path == "trajectory_memory" or row.get("operation_type", "") in {"trajectory_write", "special_token_update"}:
            source_i = int(float(source_frame)) if str(source_frame).strip() else -1
            frame_cue = frame_scores.get((seq, source_i), {})
            traj_rows.append(
                {
                    "schema": "acl2_v112tf_trajectory_memory_cue_row_v1",
                    **base,
                    "context_token_type": token_type,
                    "is_retained_keyframe": row.get("keyframe_flag", ""),
                    "trajectory_write_status": row.get("operation_type", "") == "trajectory_write",
                    "retention_status": row.get("operation_type", "") in {"retention", "budget_keep"},
                    "eviction_status": row.get("operation_type", "") in {"eviction", "budget_drop"},
                    "Q_traj_frame": frame_cue.get("Q_traj_frame", ""),
                    "semantic_lifetime_risk": frame_cue.get("semantic_lifetime_risk", ""),
                    "semantic_lifetime_support": frame_cue.get("semantic_lifetime_support", ""),
                    "g_camera": frame_cue.get("g_camera", ""),
                    "g_register": frame_cue.get("g_register", ""),
                    "g_anchor": frame_cue.get("g_anchor", ""),
                }
            )

    role_summary: dict[tuple[str, str], dict[str, Any]] = {}
    token_rows_seen = 0
    for row in iter_csv(token_path) or []:
        seq = row.get("seq_id", "")
        if seq not in SEQUENCES:
            continue
        token_rows_seen += 1
        role = row.get("semantic_role", "unknown")
        key = (seq, role)
        slot = role_summary.setdefault(
            key,
            {
                "schema": "acl2_v112tf_token_role_summary_row_v1",
                "seq": seq,
                "semantic_role": role,
                "token_count": 0,
                "semantic_trust_sum": 0.0,
                "boundary_risk_sum": 0.0,
            },
        )
        slot["token_count"] += 1
        slot["semantic_trust_sum"] += fnum(row.get("semantic_trust"), 0.0)
        slot["boundary_risk_sum"] += fnum(row.get("semantic_boundary_risk"), 0.0)
    token_role_rows = []
    for slot in role_summary.values():
        count = max(int(slot["token_count"]), 1)
        slot["semantic_trust_mean"] = float(slot["semantic_trust_sum"]) / count
        slot["boundary_risk_mean"] = float(slot["boundary_risk_sum"]) / count
        token_role_rows.append(slot)

    write_csv(STAGE2 / "anchor_memory_cue_rows.csv", anchor_rows)
    write_csv(STAGE2 / "local_window_token_cue_rows.csv", local_rows)
    write_csv(STAGE2 / "trajectory_memory_cue_rows.csv", traj_rows)
    write_csv(STAGE2 / "token_role_summary_rows.csv", token_role_rows)
    write_text(
        STAGE2 / "semantic_lifetime_policy_catalog.md",
        """# ACL2 v112TF Semantic Lifetime Policy Catalog

Pre-registered H1/T4 policies supported by existing compact context-token mask hook:

- H1_dynamic_short_stable_long
- H1_weak_medium_stable_long
- H1_boundary_register_short_anchor_long
- H1_token_type_lifetime_camera_reg_anchor
- H1_semantic_lifetime_soft_raw
- H1_semantic_lifetime_soft_znorm

Current Stage2 rows provide frame-level `g_camera`, `g_register`, `g_anchor`.
These rows are cue artifacts only; they do not claim runtime geometry.
""",
    )

    summary = {
        "schema": "acl2_v112tf_stage2_memory_specific_cue_summary_v1",
        "stage2_pass": bool(anchor_rows and local_rows and traj_rows and token_role_rows),
        "frame_semantic_rows": len(anchor_rows),
        "operation_semantic_rows_seen": op_rows_total,
        "anchor_memory_cue_rows": len(anchor_rows),
        "local_window_token_cue_rows": len(local_rows),
        "trajectory_memory_cue_rows": len(traj_rows),
        "token_semantic_rows_seen": token_rows_seen,
        "token_role_summary_rows": len(token_role_rows),
        "row_granularity_note": "local/trajectory cue rows are operation-token-group rows; full individual token rows remain available in v108 token_semantic_rows.csv",
        "sequence_frame_counts": seq_counts,
        "outputs": {
            "anchor_memory_cue_rows": rel(STAGE2 / "anchor_memory_cue_rows.csv"),
            "local_window_token_cue_rows": rel(STAGE2 / "local_window_token_cue_rows.csv"),
            "trajectory_memory_cue_rows": rel(STAGE2 / "trajectory_memory_cue_rows.csv"),
            "token_role_summary_rows": rel(STAGE2 / "token_role_summary_rows.csv"),
            "semantic_lifetime_policy_catalog": rel(STAGE2 / "semantic_lifetime_policy_catalog.md"),
            "summary": rel(STAGE2 / "memory_specific_cue_summary.json"),
        },
    }
    write_json(STAGE2 / "memory_specific_cue_summary.json", summary)
    return summary


def spearman(xs: list[float], ys: list[float]) -> float:
    pairs = [(x, y) for x, y in zip(xs, ys) if math.isfinite(x) and math.isfinite(y)]
    if len(pairs) < 3:
        return float("nan")
    xs2, ys2 = zip(*pairs)
    try:
        rx = {value: rank for rank, value in enumerate(sorted(set(xs2)), start=1)}
        ry = {value: rank for rank, value in enumerate(sorted(set(ys2)), start=1)}
        xrank = [float(rx[x]) for x in xs2]
        yrank = [float(ry[y]) for y in ys2]
        mx, my = mean(xrank), mean(yrank)
        cov = sum((x - mx) * (y - my) for x, y in zip(xrank, yrank))
        sx = math.sqrt(sum((x - mx) ** 2 for x in xrank))
        sy = math.sqrt(sum((y - my) ** 2 for y in yrank))
        return cov / (sx * sy) if sx > 0 and sy > 0 else float("nan")
    except Exception:
        return float("nan")


def build_stage25() -> dict[str, Any]:
    STAGE25.mkdir(parents=True, exist_ok=True)
    anchor_rows = read_csv(STAGE2 / "anchor_memory_cue_rows.csv")
    local_rows = read_csv(STAGE2 / "local_window_token_cue_rows.csv")
    traj_rows = read_csv(STAGE2 / "trajectory_memory_cue_rows.csv")
    a1_metrics = {
        row.get("seq", ""): fnum(row.get("full_ATE_sim3_relative_improvement_vs_baseline"))
        for row in read_csv(V111_A1 / "full_metric_rows.csv")
        if row.get("policy_id") == A1_POLICY
    }
    b1_metrics = {
        row.get("seq", ""): fnum(row.get("full_ATE_sim3_relative_improvement_vs_baseline"))
        for row in read_csv(V110_STAGE4 / "full_metric_rows.csv")
        if row.get("policy_id") == B1_POLICY
    }

    h0_rows: list[dict[str, Any]] = []
    h3_rows: list[dict[str, Any]] = []
    for seq in SEQUENCES:
        seq_anchor = [row for row in anchor_rows if row.get("seq") == seq]
        default_anchor = [row for row in seq_anchor if boolish(row.get("is_default_anchor_frame"))]
        local_seq = [row for row in local_rows if row.get("seq") == seq]
        traj_seq = [row for row in traj_rows if row.get("seq") == seq]
        anchor_dynamic = mean([fnum(row.get("dynamic_mass")) for row in default_anchor])
        anchor_boundary = mean([fnum(row.get("boundary_mass")) for row in default_anchor])
        anchor_stable = mean([fnum(row.get("stable_landmark_mass")) for row in default_anchor])
        local_dynamic = mean([fnum(row.get("dynamic_mass")) for row in local_seq])
        local_boundary = mean([fnum(row.get("boundary_mass")) for row in local_seq])
        traj_risk = mean([fnum(row.get("semantic_lifetime_risk")) for row in traj_seq])
        traj_support = mean([fnum(row.get("semantic_lifetime_support")) for row in traj_seq])
        h0_rows.append(
            {
                "schema": "acl2_v112tf_h0_influence_proxy_row_v1",
                "seq": seq,
                "anchor_dynamic_read_mass_proxy": anchor_dynamic,
                "anchor_boundary_read_mass_proxy": anchor_boundary,
                "anchor_stable_landmark_read_mass_proxy": anchor_stable,
                "local_dynamic_read_mass_proxy": local_dynamic,
                "local_boundary_read_mass_proxy": local_boundary,
                "trajectory_retained_dynamic_mass_proxy": traj_risk,
                "trajectory_retained_stable_mass_proxy": traj_support,
                "a1_full_rel": a1_metrics.get(seq, float("nan")),
                "b1_full_rel": b1_metrics.get(seq, float("nan")),
                "proxy_only": True,
                "trace_blocker": "full attention mass and head output not available; using cue/read operation proxy",
            }
        )
        h3_rows.append(
            {
                "schema": "acl2_v112tf_h3_scale_carrier_proxy_row_v1",
                "seq": seq,
                "feature_anchor_default_dynamic_mean": anchor_dynamic,
                "feature_anchor_default_stable_mean": anchor_stable,
                "feature_local_dynamic_operation_mean": local_dynamic,
                "feature_trajectory_lifetime_risk_mean": traj_risk,
                "target_a1_full_rel": a1_metrics.get(seq, float("nan")),
                "target_b1_full_rel": b1_metrics.get(seq, float("nan")),
                "feature_source": "v108 semantic cue + operation summaries",
                "proxy_only": True,
            }
        )

    feature_names = [
        "anchor_dynamic_read_mass_proxy",
        "anchor_boundary_read_mass_proxy",
        "anchor_stable_landmark_read_mass_proxy",
        "local_dynamic_read_mass_proxy",
        "local_boundary_read_mass_proxy",
        "trajectory_retained_dynamic_mass_proxy",
        "trajectory_retained_stable_mass_proxy",
    ]
    corr_rows: list[dict[str, Any]] = []
    for feature in feature_names:
        xs = [fnum(row.get(feature)) for row in h0_rows]
        for target in ("a1_full_rel", "b1_full_rel"):
            ys = [fnum(row.get(target)) for row in h0_rows]
            corr_rows.append(
                {
                    "schema": "acl2_v112tf_h0_h3_correlation_row_v1",
                    "feature": feature,
                    "target": target,
                    "spearman_rho": spearman(xs, ys),
                    "n": len([1 for x, y in zip(xs, ys) if math.isfinite(x) and math.isfinite(y)]),
                    "proxy_only": True,
                }
            )

    write_csv(STAGE25 / "h0_influence_proxy_rows.csv", h0_rows)
    write_csv(STAGE25 / "h3_scale_carrier_proxy_rows.csv", h3_rows)
    write_csv(STAGE25 / "h0_h3_proxy_correlation_rows.csv", corr_rows)
    write_text(
        STAGE25 / "H3_SCALE_CARRIER_PROBE_FEATURE_BLOCKED.md",
        """# H3_SCALE_CARRIER_PROBE_FEATURE_BLOCKED

触发条件：v112 H3 需要 trajectory camera/register/anchor hidden feature / norm / read mass，但当前 artifact 中没有完整 hidden token feature dump。

已检查文件 / function / config：
- `third_party/lingbot-map/lingbot_map/layers/attention.py`
- `results/acl2_v108tf_lingbot_memory_operation_action_surface_search/stage2_semantic_cue_bank/operation_semantic_summary.csv`
- `results/acl2_v111tf_lingbot_semantic_aware_memory_management_anchor_local_trajectory/batch_t_t2_context_token_ablation/`

失败的具体字段或指标：
缺少 hidden feature norm、per-token read mass、query-key similarity、head-wise output y_h。

当前不能继续的原因：
只能做 operation/semantic proxy probe，不能 claim full scale-carrier probe 完成。

可继续尝试的修复方向：
1. 在 SDPA path 写出 compact token norm/read-mass aggregate，不输出全 tensor。
2. 增加 per-head/per-token source mass aggregation，先 00/02 smoke。
3. 用 H0 full mass trace 修复后再做 held-out 05 probe。

哪些结论不能 claim：
不能 claim H3 完整吸收 HorizonStream Metric Readout Token 思路；当前只是 proxy-only。
""",
    )
    summary = {
        "schema": "acl2_v112tf_stage25_h0_h3_diagnostic_summary_v1",
        "stage25_pass": True,
        "h0_status": "proxy_only_partial",
        "h3_status": "proxy_only_feature_blocked",
        "h0_proxy_rows": len(h0_rows),
        "h3_proxy_rows": len(h3_rows),
        "correlation_rows": len(corr_rows),
        "top_abs_correlations": sorted(
            corr_rows,
            key=lambda row: abs(fnum(row.get("spearman_rho"), 0.0)),
            reverse=True,
        )[:5],
        "outputs": {
            "h0_influence_proxy_rows": rel(STAGE25 / "h0_influence_proxy_rows.csv"),
            "h3_scale_carrier_proxy_rows": rel(STAGE25 / "h3_scale_carrier_proxy_rows.csv"),
            "h0_h3_proxy_correlation_rows": rel(STAGE25 / "h0_h3_proxy_correlation_rows.csv"),
            "h3_feature_blocker": rel(STAGE25 / "H3_SCALE_CARRIER_PROBE_FEATURE_BLOCKED.md"),
            "summary": rel(STAGE25 / "stage25_h0_h3_summary.json"),
        },
    }
    write_json(STAGE25 / "stage25_h0_h3_summary.json", summary)
    return summary


def main() -> int:
    summary = {
        "stage0": build_stage0(),
        "stage1": build_stage1(),
        "stage2": build_stage2(),
        "stage25": build_stage25(),
    }
    write_json(RESULT_ROOT / "v112_stage0_1_2_25_audit_summary.json", summary)
    print(json.dumps(clean_json(summary), indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
