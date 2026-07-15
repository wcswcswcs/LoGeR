#!/usr/bin/env python3
"""Build ACL2 v110R Stage0 evidence-freeze artifacts.

Stage0 is read-only with respect to upstream experiments. It freezes the
current v105/v107/v108/v109 evidence boundary before any v110R model run.
"""

from __future__ import annotations

import csv
import json
import math
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "results/acl2_v110r_lingbot_multisurface_full_ate_boost_semantic_causality"
OUT = RESULT_ROOT / "stage0_evidence_freeze"

V105 = ROOT / "results/acl2_v105tf_dualtrack_lingbot_loger_evidence_eligibility"
V107TF = ROOT / "results/acl2_v107tf_lingbot_cache_operation_observability_semantic_aware_update_retention"
V107R = ROOT / "results/acl2_v107r_lingbot_semantic_memory_decision_cue_operation_control"
V108 = ROOT / "results/acl2_v108tf_lingbot_memory_operation_action_surface_search"
V109 = ROOT / "results/acl2_v109tf_lingbot_f_surface_causal_dissection_semantic_memory_control"

V108_STAGE1 = V108 / "stage1_action_surface_contract"
V108_STAGE2 = V108 / "stage2_semantic_cue_bank"
V108_STAGE5 = V108 / "stage5_full_kitti_00_01_02_05_validation"
V109_F19 = V109 / "stage2_role_specific_safety_candidates"
V109_F19_CONTROLS = V109 / "stage2_f19_keyframe_controls"

EXPECTED_BASELINE = {
    "00": 46.00057328153847,
    "01": 57.097417656974685,
    "02": 77.48077587398916,
    "05": 19.961256567907505,
}
SEQUENCES = ("00", "01", "02", "05")
F19_POLICY = "F19_dynamic_or_special_admitted_high_risk_else_weak_context"


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


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
    path.write_text(json.dumps(clean_json(data), indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


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


def process_rows() -> list[str]:
    proc = subprocess.run(
        ["ps", "-eo", "pid,ppid,stat,etime,cmd"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    markers = (
        "third_party/lingbot-map/benchmark",
        "run_worker.py",
        "run_v108tf_gpu_serial_policy_manifest.py",
        "run_v107r_stage6_semantic_wrapper_policy_manifest.py",
        "ACL2_V105_STAGE4_ACTION",
        "ACL2_V108_STAGE4_POLICY_ID",
    )
    self_markers = (
        "build_v110r_stage0_evidence_freeze.py",
        "ps -eo pid,ppid,stat,etime,cmd",
        "rg ",
    )
    rows: list[str] = []
    for line in proc.stdout.splitlines():
        if not any(marker in line for marker in markers):
            continue
        if any(marker in line for marker in self_markers):
            continue
        rows.append(line.strip())
    return rows


def artifact_manifest() -> list[dict[str, Any]]:
    artifacts = [
        ("v105_baseline_full_metrics", V105 / "stage1_lingbot_baseline/full_sequence_metrics/lingbot_stream_default_full_metrics.csv", "required"),
        ("v107tf_operation_trace_summary", V107TF / "stage1_cache_operation_instrumentation/operation_trace_summary.json", "required"),
        ("v107tf_operation_trace_rows", V107TF / "stage1_cache_operation_instrumentation/operation_trace_rows.csv", "required"),
        ("v107tf_operation_trace_parity_rows", V107TF / "stage1_cache_operation_instrumentation/operation_trace_parity_rows.csv", "required"),
        ("v107r_semantic_cue_summary", V107R / "stage1_semantic_cue_bank/semantic_cue_summary.json", "required"),
        ("v107r_frame_semantic_summary", V107R / "stage1_semantic_cue_bank/frame_semantic_summary.csv", "required"),
        ("v108_stage1_surface_feasibility", V108_STAGE1 / "action_surface_implementation_feasibility.csv", "required"),
        ("v108_stage1_surface_contract", V108_STAGE1 / "action_surface_contract.md", "required"),
        ("v108_stage2_semantic_cue_bank_summary", V108_STAGE2 / "stage2_summary.json", "required"),
        ("v108_stage2_frame_semantic_summary", V108_STAGE2 / "frame_semantic_summary.csv", "required"),
        ("v108_stage2_operation_semantic_summary", V108_STAGE2 / "operation_semantic_summary.csv", "required"),
        ("v108_stage5_summary", V108_STAGE5 / "stage5_summary.json", "required"),
        ("v108_stage5_action_config_rows", V108_STAGE5 / "action_config_rows.csv", "required"),
        ("v108_stage5_full_metric_rows", V108_STAGE5 / "full_sequence_metric_rows.csv", "required"),
        ("v108_stage5_semantic_control_rows", V108_STAGE5 / "semantic_control_rows.csv", "required"),
        ("v109_stage2_f_core_summary", V109 / "stage2_f_core_ablation/stage2_summary.json", "required"),
        ("v109_f19_summary", V109_F19 / "role_specific_safety_candidate_summary.json", "required"),
        ("v109_f19_full_metric_rows", V109_F19 / "full_metric_rows.csv", "required"),
        ("v109_f19_action_config_rows", V109_F19 / "action_config_rows.csv", "required"),
        ("v109_f19_keyframe_control_summary", V109_F19_CONTROLS / "f19_keyframe_control_summary.json", "required"),
        ("v109_f19_keyframe_control_rows", V109_F19_CONTROLS / "full_metric_rows.csv", "required"),
        ("v109_semantic_content_not_causal_yet", V109 / "SEMANTIC_CONTENT_NOT_CAUSAL_YET.md", "required"),
    ]
    rows: list[dict[str, Any]] = []
    for artifact_id, path, requirement in artifacts:
        row_count: int | str = ""
        if path.exists() and path.suffix == ".csv":
            row_count = len(read_csv(path))
        rows.append(
            {
                "schema": "acl2_v110r_stage0_artifact_manifest_row_v1",
                "artifact_id": artifact_id,
                "path": rel(path),
                "requirement": requirement,
                "exists": path.exists(),
                "suffix": path.suffix,
                "row_count": row_count,
            }
        )
    return rows


def frozen_baseline_table() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    src = V105 / "stage1_lingbot_baseline/full_sequence_metrics/lingbot_stream_default_full_metrics.csv"
    rows: list[dict[str, Any]] = []
    exact: dict[str, Any] = {}
    for row in read_csv(src):
        seq = row.get("seq", "")
        if seq not in EXPECTED_BASELINE:
            continue
        observed = fnum(row.get("ATE_full_sim3_m"))
        expected = EXPECTED_BASELINE[seq]
        abs_diff = abs(observed - expected) if math.isfinite(observed) else float("nan")
        exact[seq] = {
            "observed": observed,
            "expected": expected,
            "abs_diff": abs_diff,
            "match": math.isfinite(abs_diff) and abs_diff <= 1e-12,
        }
        rows.append(
            {
                "schema": "acl2_v110r_stage0_frozen_baseline_row_v1",
                "seq": seq,
                "dataset": row.get("dataset", ""),
                "method": row.get("method", ""),
                "frames": row.get("frames", ""),
                "ATE_full_sim3_m": row.get("ATE_full_sim3_m", ""),
                "expected_ATE_full_sim3_m": expected,
                "baseline_abs_diff": abs_diff,
                "baseline_exact_match": math.isfinite(abs_diff) and abs_diff <= 1e-12,
                "benchmark_rpe_trans": row.get("benchmark_rpe_trans", ""),
                "benchmark_rpe_rot": row.get("benchmark_rpe_rot", ""),
                "final_error_m": row.get("final_error_m", ""),
                "rolling_ATE_mean": row.get("rolling_ATE_mean", ""),
                "rolling_ATE_p90": row.get("rolling_ATE_p90", ""),
                "rolling_ATE_max": row.get("rolling_ATE_max", ""),
                "rolling_worse_fraction_gt_0p05": row.get("rolling_worse_fraction_gt_0p05", ""),
                "full_global_sim3_scale": row.get("full_global_sim3_scale", ""),
                "full_global_sim3_yaw_rad": row.get("full_global_sim3_yaw_rad", ""),
                "local_window_ATE_median": row.get("local_window_ATE_median", ""),
                "adjacent_log_scale_jump_median": row.get("adjacent_log_scale_jump_median", ""),
                "handoff_transfer_penalty_median": row.get("handoff_transfer_penalty_median", ""),
                "source": rel(src),
            }
        )
    return rows, exact


def f19_champion_rows() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    summary = read_json(V109_F19 / "role_specific_safety_candidate_summary.json")
    rows: list[dict[str, Any]] = []
    rels: list[float] = []
    for row in read_csv(V109_F19 / "full_metric_rows.csv"):
        if row.get("policy_id") != F19_POLICY:
            continue
        rel_imp = fnum(row.get("full_ATE_sim3_relative_improvement_vs_baseline"))
        rels.append(rel_imp)
        rows.append(
            {
                "schema": "acl2_v110r_stage0_f19_champion_metric_row_v1",
                "surface_id": "F",
                "policy_id": row.get("policy_id", ""),
                "policy_family": row.get("policy_family", ""),
                "seq": row.get("seq", ""),
                "baseline_full_ATE_sim3": row.get("baseline_full_ATE_sim3", ""),
                "full_ATE_sim3": row.get("full_ATE_sim3", ""),
                "full_ATE_sim3_delta_action_minus_baseline": row.get("full_ATE_sim3_delta_action_minus_baseline", ""),
                "full_ATE_sim3_relative_improvement_vs_baseline": row.get("full_ATE_sim3_relative_improvement_vs_baseline", ""),
                "full_RPE_translation": row.get("full_RPE_translation", ""),
                "full_RPE_rotation": row.get("full_RPE_rotation", ""),
                "final_error_m": row.get("final_error_m", ""),
                "final_error_relative_improvement_vs_baseline": row.get("final_error_relative_improvement_vs_baseline", ""),
                "global_sim3_scale": row.get("global_sim3_scale", ""),
                "global_sim3_yaw_rad": row.get("global_sim3_yaw_rad", ""),
                "runtime_sec": row.get("runtime_sec", ""),
                "local_window_ATE_rel_improvement_vs_baseline_median": row.get("local_window_ATE_rel_improvement_vs_baseline_median", ""),
                "adjacent_log_scale_jump_p90": row.get("adjacent_log_scale_jump_p90", ""),
                "handoff_transfer_penalty_p90": row.get("handoff_transfer_penalty_p90", ""),
                "metric_available": row.get("metric_available", ""),
                "action_fidelity_pass": row.get("action_fidelity_pass", ""),
                "source": rel(V109_F19 / "full_metric_rows.csv"),
            }
        )
    aggregate = {
        "row_count": len(rows),
        "median_full_rel": median(rels),
        "mean_full_rel": mean(rels),
        "improved_seq_count": sum(1 for v in rels if math.isfinite(v) and v > 0.0),
        "max_harm": max_harm(rels),
        "summary_taxonomy": summary.get("taxonomy", ""),
        "summary_safety_candidate_pass": summary.get("safety_candidate_pass", False),
        "summary_metric_complete": summary.get("metric_complete", False),
        "summary_all_action_fidelity": summary.get("all_action_fidelity", False),
    }
    return rows, aggregate


def f19_control_rows() -> list[dict[str, Any]]:
    summary = read_json(V109_F19_CONTROLS / "f19_keyframe_control_summary.json")
    rows = []
    for row in read_csv(V109_F19_CONTROLS / "f19_keyframe_control_summary_rows.csv"):
        rows.append(
            {
                "schema": "acl2_v110r_stage0_f19_keyframe_control_row_v1",
                **row,
                "stage0_summary_taxonomy": summary.get("taxonomy", ""),
                "stage0_summary_blocker": summary.get("blocker", ""),
                "stage0_supports_f19_causality": summary.get("f19_keyframe_control_supports_f19_causality", ""),
                "source": rel(V109_F19_CONTROLS / "f19_keyframe_control_summary_rows.csv"),
            }
        )
    return rows


def v108_surface_summary_rows() -> list[dict[str, Any]]:
    contract = {
        row.get("surface_id", ""): row
        for row in read_csv(V108_STAGE1 / "action_surface_implementation_feasibility.csv")
    }
    grouped: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in read_csv(V108_STAGE5 / "full_sequence_metric_rows.csv"):
        key = (row.get("surface_id", ""), row.get("policy_id", ""), row.get("policy_family", ""))
        grouped[key].append(row)
    rows: list[dict[str, Any]] = []
    for (surface, policy_id, family), metric_rows in sorted(grouped.items()):
        rels = [fnum(row.get("full_ATE_sim3_relative_improvement_vs_baseline")) for row in metric_rows]
        local_rels = [fnum(row.get("local_window_ATE_rel_improvement_vs_baseline_median")) for row in metric_rows]
        row_contract = contract.get(surface, {})
        rows.append(
            {
                "schema": "acl2_v110r_stage0_v108_surface_summary_row_v1",
                "surface_id": surface,
                "policy_id": policy_id,
                "policy_family": family,
                "sequence_count": len(metric_rows),
                "median_full_rel": median(rels),
                "mean_full_rel": mean(rels),
                "improved_seq_count": sum(1 for v in rels if math.isfinite(v) and v > 0.0),
                "max_harm": max_harm(rels),
                "median_local_window_rel": median(local_rels),
                "all_metric_available": all(boolish(row.get("metric_available")) for row in metric_rows),
                "all_action_fidelity": all(boolish(row.get("action_fidelity_pass")) for row in metric_rows),
                "operation_type": row_contract.get("operation_type", ""),
                "implementation_status": row_contract.get("implementation_status", ""),
                "has_existing_runtime_knob": row_contract.get("has_existing_runtime_knob", ""),
                "new_hook_needed": row_contract.get("new_hook_needed", ""),
                "full_sequence_pilot_allowed": row_contract.get("full_sequence_pilot_allowed", ""),
                "contract_note": row_contract.get("note", ""),
                "source": rel(V108_STAGE5 / "full_sequence_metric_rows.csv"),
            }
        )
    return rows


def allowed_action_surface_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in read_csv(V108_STAGE1 / "action_surface_implementation_feasibility.csv"):
        surface = row.get("surface_id", "")
        allowed = boolish(row.get("full_sequence_pilot_allowed"))
        if surface in {"C", "D"}:
            v110_status = "hook_audit_required_before_claim"
        elif allowed:
            v110_status = "stage2_abef_candidate_allowed"
        else:
            v110_status = "not_allowed_without_new_hook"
        rows.append(
            {
                "schema": "acl2_v110r_stage0_allowed_action_surface_row_v1",
                "surface_id": surface,
                "operation_type": row.get("operation_type", ""),
                "implementation_status": row.get("implementation_status", ""),
                "has_existing_runtime_knob": row.get("has_existing_runtime_knob", ""),
                "new_hook_needed": row.get("new_hook_needed", ""),
                "full_sequence_pilot_allowed": row.get("full_sequence_pilot_allowed", ""),
                "v110_status": v110_status,
                "v110_priority": "1" if surface in {"A", "B", "E", "F"} else "2",
                "claim_boundary": "diagnostic_or_hook_smoke_only" if surface in {"C", "D"} else "full_00_02_pilot_required",
                "note": row.get("note", ""),
            }
        )
    return rows


def v107_availability_rows() -> list[dict[str, Any]]:
    v107tf = read_json(V107TF / "stage1_cache_operation_instrumentation/operation_trace_summary.json")
    v107r = read_json(V107R / "stage1_semantic_cue_bank/semantic_cue_summary.json")
    v108_stage2 = read_json(V108_STAGE2 / "stage2_summary.json")
    return [
        {
            "schema": "acl2_v110r_stage0_trace_or_cue_availability_row_v1",
            "source_id": "v107tf_operation_trace",
            "path": rel(V107TF / "stage1_cache_operation_instrumentation/operation_trace_summary.json"),
            "available": bool(v107tf),
            "pass_flag": v107tf.get("trace_parity_pass", ""),
            "operation_row_count": v107tf.get("operation_row_count", ""),
            "observed_operation_types": ";".join(v107tf.get("observed_operation_types", [])),
            "note": "trace-visible memory operation evidence; C/D still hook-needed for runtime claim",
        },
        {
            "schema": "acl2_v110r_stage0_trace_or_cue_availability_row_v1",
            "source_id": "v107r_semantic_cue_bank",
            "path": rel(V107R / "stage1_semantic_cue_bank/semantic_cue_summary.json"),
            "available": bool(v107r),
            "pass_flag": v107r.get("stage1_pass", ""),
            "semantic_projection_coverage": v107r.get("semantic_projection_coverage", ""),
            "semantic_patch_nonvoid_ratio": v107r.get("semantic_patch_nonvoid_ratio", ""),
            "semantic_patch_purity_mean": v107r.get("semantic_patch_purity_mean", ""),
            "note": "prior semantic cue bank evidence",
        },
        {
            "schema": "acl2_v110r_stage0_trace_or_cue_availability_row_v1",
            "source_id": "v108_semantic_cue_bank",
            "path": rel(V108_STAGE2 / "stage2_summary.json"),
            "available": bool(v108_stage2),
            "pass_flag": v108_stage2.get("stage2_pass", ""),
            "frame_semantic_coverage": v108_stage2.get("frame_semantic_coverage", ""),
            "operation_rows_join_coverage_mean": v108_stage2.get("operation_rows_join_coverage_mean", ""),
            "token_semantic_row_count": v108_stage2.get("token_semantic_row_count", ""),
            "operation_row_count": v108_stage2.get("operation_row_count", ""),
            "note": "v110 primary semantic/operation cue source",
        },
    ]


def forbidden_repeat_text() -> str:
    return """# ACL2 v110R Forbidden Repeat List

1. Do not claim geometry improvement without full KITTI ATE.
2. Do not use selected-window, 96F, local L3, or trace movement as a full-geometry substitute.
3. Do not claim semantic-aware method if semantic shuffle / role rotation / same-count / same-bucket controls match the effect.
4. Do not treat F19 as a final semantic method; it is the current champion and safety baseline only.
5. Do not promote E-surface high-gain results without 01/05 hard-negative safety.
6. Do not claim C/D runtime behavior without a real hook, no-action parity, and action fidelity trace.
7. Do not use GT, external depth, MoGe, SLAM, or post-hoc Sim3 as runtime cues.
8. Do not hand-tune per-sequence policies after seeing full ATE.
"""


def no_claim_boundary_text(f19_aggregate: dict[str, Any], f19_control: dict[str, Any]) -> str:
    return f"""# ACL2 v110R No-Claim Boundary

F19 is the current champion baseline:

```text
median_full_rel={f19_aggregate.get('median_full_rel')}
mean_full_rel={f19_aggregate.get('mean_full_rel')}
improved_seq_count={f19_aggregate.get('improved_seq_count')}
max_harm={f19_aggregate.get('max_harm')}
```

F19 is not a semantic-aware method claim because v109 exact-count keyframe controls reported:

```text
taxonomy={f19_control.get('taxonomy')}
blocker={f19_control.get('blocker')}
f19_keyframe_control_supports_f19_causality={f19_control.get('f19_keyframe_control_supports_f19_causality')}
best_same_seq_control_match_f19_count={f19_control.get('best_same_seq_control_match_f19_count')}
```

v110R may use F19 as:

```text
current champion
safety baseline
component in limited combination search
strong keyframe/cache schedule baseline
```

v110R may not use F19 as:

```text
final semantic-aware method
proof that semantic content is causal
replacement for multi-surface full ATE candidate search
```
"""


def build() -> dict[str, Any]:
    OUT.mkdir(parents=True, exist_ok=True)
    manifest_rows = artifact_manifest()
    baseline_rows, baseline_exact = frozen_baseline_table()
    f19_rows, f19_aggregate = f19_champion_rows()
    f19_control_summary = read_json(V109_F19_CONTROLS / "f19_keyframe_control_summary.json")
    f19_control_summary_rows = f19_control_rows()
    surface_rows = v108_surface_summary_rows()
    allowed_rows = allowed_action_surface_rows()
    v107_rows = v107_availability_rows()
    pending = process_rows()

    required_missing = [
        row["artifact_id"]
        for row in manifest_rows
        if row["requirement"] == "required" and not boolish(row["exists"])
    ]
    baseline_exact_match = (
        set(baseline_exact) == set(SEQUENCES)
        and all(bool(row.get("match")) for row in baseline_exact.values())
    )
    f19_champion_readable = (
        len(f19_rows) == 4
        and bool(f19_aggregate.get("summary_safety_candidate_pass"))
        and bool(f19_aggregate.get("summary_metric_complete"))
        and bool(f19_aggregate.get("summary_all_action_fidelity"))
    )
    v108_action_configs_readable = len(read_csv(V108_STAGE5 / "action_config_rows.csv")) > 0
    v109_action_configs_readable = (
        len(read_csv(V109_F19 / "action_config_rows.csv")) > 0
        and len(read_csv(V109_F19_CONTROLS / "action_config_rows.csv")) > 0
    )
    v108_v109_action_configs_readable = v108_action_configs_readable and v109_action_configs_readable
    roots_present = all(
        path.exists()
        for path in [
            V105,
            V107TF,
            V107R,
            V108,
            V109,
            V108_STAGE1,
            V108_STAGE2,
            V108_STAGE5,
            V109_F19,
            V109_F19_CONTROLS,
        ]
    )
    no_stale_lingbot_worker = not pending
    stage0_pass = (
        baseline_exact_match
        and f19_champion_readable
        and v108_v109_action_configs_readable
        and roots_present
        and no_stale_lingbot_worker
        and not required_missing
    )
    blockers: list[str] = []
    if not baseline_exact_match:
        blockers.append("baseline_exact_match_failed")
    if not f19_champion_readable:
        blockers.append("f19_champion_metrics_not_readable")
    if not v108_v109_action_configs_readable:
        blockers.append("v108_v109_action_configs_not_readable")
    if not roots_present:
        blockers.append("required_artifact_root_missing")
    if not no_stale_lingbot_worker:
        blockers.append("stale_lingbot_worker_running")
    blockers.extend(f"missing_required_artifact:{item}" for item in required_missing)

    write_csv(OUT / "available_artifact_manifest.csv", manifest_rows)
    write_csv(OUT / "frozen_baseline_table.csv", baseline_rows)
    write_csv(OUT / "f19_champion_metrics.csv", f19_rows)
    write_csv(OUT / "f19_keyframe_control_summary_rows.csv", f19_control_summary_rows)
    write_csv(OUT / "v108_surface_summary.csv", surface_rows)
    write_csv(OUT / "allowed_action_surfaces.csv", allowed_rows)
    write_csv(OUT / "trace_cue_availability.csv", v107_rows)
    write_text(OUT / "forbidden_repeat_list.md", forbidden_repeat_text())
    write_text(OUT / "no_claim_boundary.md", no_claim_boundary_text(f19_aggregate, f19_control_summary))

    summary = {
        "schema": "acl2_v110r_stage0_evidence_freeze_summary_v1",
        "stage0_pass": stage0_pass,
        "blockers": blockers,
        "baseline_exact_match": baseline_exact_match,
        "baseline_exact_match_by_seq": baseline_exact,
        "f19_champion_metrics_readable": f19_champion_readable,
        "f19_champion_policy_id": F19_POLICY,
        "f19_champion_aggregate": f19_aggregate,
        "f19_control_taxonomy": f19_control_summary.get("taxonomy", ""),
        "f19_control_blocker": f19_control_summary.get("blocker", ""),
        "f19_control_supports_semantic_causality": f19_control_summary.get("f19_keyframe_control_supports_f19_causality", False),
        "v108_action_configs_readable": v108_action_configs_readable,
        "v109_action_configs_readable": v109_action_configs_readable,
        "v108_v109_action_configs_readable": v108_v109_action_configs_readable,
        "required_roots_present": roots_present,
        "required_missing_artifacts": required_missing,
        "pending_lingbot_process_rows": pending,
        "no_stale_lingbot_worker": no_stale_lingbot_worker,
        "allowed_action_surfaces": {
            row["surface_id"]: row["v110_status"]
            for row in allowed_rows
        },
        "stage0_boundary": {
            "F19": "current_champion_but_not_final_semantic_method",
            "E": "high_gain_high_risk_control",
            "A_B": "schedule_baselines_or_low_risk_candidates",
            "C_D": "hook_audit_required_before_claim",
            "F": "promising_but_semantic_causality_unresolved",
        },
        "outputs": {
            "stage0_summary": rel(OUT / "stage0_summary.json"),
            "available_artifact_manifest": rel(OUT / "available_artifact_manifest.csv"),
            "frozen_baseline_table": rel(OUT / "frozen_baseline_table.csv"),
            "f19_champion_metrics": rel(OUT / "f19_champion_metrics.csv"),
            "f19_keyframe_control_summary_rows": rel(OUT / "f19_keyframe_control_summary_rows.csv"),
            "v108_surface_summary": rel(OUT / "v108_surface_summary.csv"),
            "allowed_action_surfaces": rel(OUT / "allowed_action_surfaces.csv"),
            "trace_cue_availability": rel(OUT / "trace_cue_availability.csv"),
            "forbidden_repeat_list": rel(OUT / "forbidden_repeat_list.md"),
            "no_claim_boundary": rel(OUT / "no_claim_boundary.md"),
        },
    }
    write_json(OUT / "stage0_summary.json", summary)
    if not stage0_pass:
        write_text(
            OUT / "STAGE0_EVIDENCE_FREEZE_BLOCKED.md",
            "# ACL2 v110R Stage0 Evidence Freeze Blocked\n\n"
            f"blockers: `{blockers}`\n\n"
            f"pending_lingbot_process_rows: `{pending}`\n",
        )
    return summary


def main() -> None:
    print(json.dumps(clean_json(build()), indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
