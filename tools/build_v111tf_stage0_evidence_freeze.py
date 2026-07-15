#!/usr/bin/env python3
"""Build ACL2 v111TF Stage0 evidence-freeze artifacts.

Stage0 is deliberately read-only with respect to upstream experiments. It
freezes the v108/v109/v110R LingBot evidence boundary before v111 action
experiments.
"""

from __future__ import annotations

import csv
import json
import math
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "results/acl2_v111tf_lingbot_semantic_aware_memory_management_anchor_local_trajectory"
OUT = RESULT_ROOT / "stage0_evidence_freeze"

V105 = ROOT / "results/acl2_v105tf_dualtrack_lingbot_loger_evidence_eligibility"
V108 = ROOT / "results/acl2_v108tf_lingbot_memory_operation_action_surface_search"
V109 = ROOT / "results/acl2_v109tf_lingbot_f_surface_causal_dissection_semantic_memory_control"
V110 = ROOT / "results/acl2_v110r_lingbot_multisurface_full_ate_boost_semantic_causality"

V108_STAGE1 = V108 / "stage1_action_surface_contract"
V108_STAGE2 = V108 / "stage2_semantic_cue_bank"
V109_F19 = V109 / "stage2_role_specific_safety_candidates"
V109_F19_CONTROLS = V109 / "stage2_f19_keyframe_controls"
V110_STAGE0 = V110 / "stage0_evidence_freeze"
V110_STAGE4 = V110 / "stage4_full_00_01_02_05_validation"
V110_FINAL = V110 / "final_decision"

SEQUENCES = ("00", "01", "02", "05")
EXPECTED_BASELINE = {
    "00": 46.00057328153847,
    "01": 57.097417656974685,
    "02": 77.48077587398916,
    "05": 19.9612565679075,
}
F19_POLICY = "F19_dynamic_or_special_admitted_high_risk_else_weak_context"
B1_POLICIES = {"B1_semantic_only", "B1_semantic_plus_internal"}


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
        "run_v108tf_gpu_serial_policy_manifest.py",
        "run_v107r_stage6_semantic_wrapper_policy_manifest.py",
        "ACL2_V108_STAGE4_POLICY_ID",
        "ACL2_V111",
    )
    self_markers = (
        "build_v111tf_stage0_evidence_freeze.py",
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
        ("v105_full_kitti_baseline", V105 / "stage1_lingbot_baseline/full_sequence_metrics/lingbot_stream_default_full_metrics.csv", "required"),
        ("v108_stage1_action_surface_contract", V108_STAGE1 / "action_surface_contract.md", "required"),
        ("v108_stage1_feasibility", V108_STAGE1 / "action_surface_implementation_feasibility.csv", "required"),
        ("v108_stage2_semantic_summary", V108_STAGE2 / "stage2_summary.json", "required"),
        ("v108_stage2_frame_semantic_summary", V108_STAGE2 / "frame_semantic_summary.csv", "required"),
        ("v108_stage2_operation_semantic_summary", V108_STAGE2 / "operation_semantic_summary.csv", "required"),
        ("v108_stage2_token_semantic_rows", V108_STAGE2 / "token_semantic_rows.csv", "required"),
        ("v109_f19_summary", V109_F19 / "role_specific_safety_candidate_summary.json", "required"),
        ("v109_f19_full_metric_rows", V109_F19 / "full_metric_rows.csv", "required"),
        ("v109_f19_action_fidelity_rows", V109_F19 / "action_fidelity_rows.csv", "required"),
        ("v109_f19_keyframe_control_summary", V109_F19_CONTROLS / "f19_keyframe_control_summary.json", "required"),
        ("v109_f19_keyframe_control_rows", V109_F19_CONTROLS / "f19_keyframe_control_summary_rows.csv", "required"),
        ("v110_stage0_summary", V110_STAGE0 / "stage0_summary.json", "required"),
        ("v110_stage4_full_metric_rows", V110_STAGE4 / "full_metric_rows.csv", "required"),
        ("v110_stage4_action_fidelity_rows", V110_STAGE4 / "action_fidelity_rows.csv", "required"),
        ("v110_stage4_semantic_control_rows", V110_STAGE4 / "semantic_control_rows.csv", "required"),
        ("v110_stage4_summary", V110_STAGE4 / "stage4_summary.json", "required"),
        ("v110_final_decision", V110_FINAL / "final_decision.json", "required"),
        ("lingbot_wrapper", ROOT / "third_party/lingbot-map/benchmark/methods/lingbot_map.py", "required"),
        ("lingbot_stream_aggregator", ROOT / "third_party/lingbot-map/lingbot_map/aggregator/stream.py", "required"),
        ("lingbot_stream_window_model", ROOT / "third_party/lingbot-map/lingbot_map/models/gct_stream_window.py", "required"),
        ("lingbot_stream_model", ROOT / "third_party/lingbot-map/lingbot_map/models/gct_stream.py", "required"),
        ("lingbot_attention_layer", ROOT / "third_party/lingbot-map/lingbot_map/layers/attention.py", "required"),
    ]
    rows: list[dict[str, Any]] = []
    for artifact_id, path, requirement in artifacts:
        row_count: int | str = ""
        size_bytes: int | str = path.stat().st_size if path.exists() else ""
        if path.exists() and path.suffix == ".csv" and size_bytes != "" and int(size_bytes) <= 50_000_000:
            row_count = len(read_csv(path))
        elif path.exists() and path.suffix == ".csv":
            row_count = "skipped_large_csv"
        rows.append(
            {
                "schema": "acl2_v111tf_stage0_artifact_manifest_row_v1",
                "artifact_id": artifact_id,
                "path": rel(path),
                "requirement": requirement,
                "exists": path.exists(),
                "suffix": path.suffix,
                "size_bytes": size_bytes,
                "row_count": row_count,
            }
        )
    return rows


def baseline_table() -> tuple[list[dict[str, Any]], dict[str, Any]]:
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
                "schema": "acl2_v111tf_stage0_full_kitti_baseline_row_v1",
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
                "source": rel(src),
            }
        )
    return rows, exact


def b1_rows() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    src = V110_STAGE4 / "full_metric_rows.csv"
    rows: list[dict[str, Any]] = []
    rels_by_policy: dict[str, list[float]] = {policy: [] for policy in B1_POLICIES}
    for row in read_csv(src):
        if row.get("policy_id") not in B1_POLICIES:
            continue
        rel_imp = fnum(row.get("full_ATE_sim3_relative_improvement_vs_baseline"))
        rels_by_policy[row.get("policy_id", "")].append(rel_imp)
        rows.append(
            {
                "schema": "acl2_v111tf_stage0_b1_champion_metric_row_v1",
                "candidate_id": row.get("candidate_id", ""),
                "surface_id": row.get("surface_id", ""),
                "policy_id": row.get("policy_id", ""),
                "policy_family": row.get("policy_family", ""),
                "seq": row.get("seq", ""),
                "full_ATE_sim3": row.get("full_ATE_sim3", ""),
                "baseline_full_ATE_sim3": row.get("baseline_full_ATE_sim3", ""),
                "full_ATE_sim3_delta_action_minus_baseline": row.get("full_ATE_sim3_delta_action_minus_baseline", ""),
                "full_ATE_sim3_relative_improvement_vs_baseline": row.get("full_ATE_sim3_relative_improvement_vs_baseline", ""),
                "full_RPE_translation": row.get("full_RPE_translation", ""),
                "full_RPE_rotation": row.get("full_RPE_rotation", ""),
                "final_error_m": row.get("final_error_m", ""),
                "final_error_relative_improvement_vs_baseline": row.get("final_error_relative_improvement_vs_baseline", ""),
                "global_sim3_scale": row.get("global_sim3_scale", ""),
                "global_sim3_yaw_rad": row.get("global_sim3_yaw_rad", ""),
                "rolling_p90_source": rel(V110_STAGE4 / "policy_summary_rows.csv"),
                "local_window_ATE_rel_improvement_vs_baseline_median": row.get("local_window_ATE_rel_improvement_vs_baseline_median", ""),
                "adjacent_log_scale_jump_p90": row.get("adjacent_log_scale_jump_p90", ""),
                "handoff_transfer_penalty_p90": row.get("handoff_transfer_penalty_p90", ""),
                "runtime_sec": row.get("runtime_sec", ""),
                "peak_gpu_memory_mb": row.get("peak_gpu_memory_mb", ""),
                "metric_available": row.get("metric_available", ""),
                "all_phase_success": row.get("all_phase_success", ""),
                "action_fidelity_pass": row.get("action_fidelity_pass", ""),
                "source": rel(src),
            }
        )
    semantic = read_csv(V110_STAGE4 / "semantic_control_rows.csv")
    semantic_b1 = next((row for row in semantic if row.get("candidate_id") == "B1"), {})
    final = read_json(V110_FINAL / "final_decision.json")
    primary = final.get("primary_candidate", {})
    aggregate = {
        "primary_policy_id": final.get("primary_candidate_policy_id", "B1_semantic_only"),
        "median_full_rel": primary.get("median_full_rel", median(rels_by_policy.get("B1_semantic_only", []))),
        "mean_full_rel": primary.get("mean_full_rel", mean(rels_by_policy.get("B1_semantic_only", []))),
        "improved_seq_count": primary.get("improved_seq_count", sum(1 for v in rels_by_policy.get("B1_semantic_only", []) if math.isfinite(v) and v > 0)),
        "max_harm": primary.get("max_harm", max_harm(rels_by_policy.get("B1_semantic_only", []))),
        "rolling_p90_median_rel": primary.get("rolling_p90_median_rel"),
        "final_error_median_rel": primary.get("final_error_median_rel"),
        "local_window_median_harm": primary.get("local_window_median_harm"),
        "semantic_plus_internal_median": fnum(semantic_b1.get("semantic_plus_median_full_rel")),
        "semantic_only_median": fnum(semantic_b1.get("semantic_only_median_full_rel")),
        "semantic_plus_minus_semantic_only_median": fnum(semantic_b1.get("semantic_plus_minus_semantic_only_median")),
        "semantic_causality_claim_allowed": boolish(semantic_b1.get("semantic_causality_claim_allowed")),
        "final_taxonomy": final.get("final_taxonomy", ""),
    }
    return rows, aggregate


def f19_rows() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    src = V109_F19 / "full_metric_rows.csv"
    rows: list[dict[str, Any]] = []
    rels: list[float] = []
    for row in read_csv(src):
        if row.get("policy_id") != F19_POLICY:
            continue
        rel_imp = fnum(row.get("full_ATE_sim3_relative_improvement_vs_baseline"))
        rels.append(rel_imp)
        rows.append(
            {
                "schema": "acl2_v111tf_stage0_f19_reference_metric_row_v1",
                "surface_id": "F",
                "policy_id": row.get("policy_id", ""),
                "policy_family": row.get("policy_family", ""),
                "seq": row.get("seq", ""),
                "full_ATE_sim3": row.get("full_ATE_sim3", ""),
                "baseline_full_ATE_sim3": row.get("baseline_full_ATE_sim3", ""),
                "full_ATE_sim3_relative_improvement_vs_baseline": row.get("full_ATE_sim3_relative_improvement_vs_baseline", ""),
                "final_error_relative_improvement_vs_baseline": row.get("final_error_relative_improvement_vs_baseline", ""),
                "global_sim3_scale": row.get("global_sim3_scale", ""),
                "global_sim3_yaw_rad": row.get("global_sim3_yaw_rad", ""),
                "local_window_ATE_rel_improvement_vs_baseline_median": row.get("local_window_ATE_rel_improvement_vs_baseline_median", ""),
                "metric_available": row.get("metric_available", ""),
                "action_fidelity_pass": row.get("action_fidelity_pass", ""),
                "source": rel(src),
            }
        )
    safety = read_json(V109_F19 / "role_specific_safety_candidate_summary.json")
    controls = read_json(V109_F19_CONTROLS / "f19_keyframe_control_summary.json")
    aggregate = {
        "policy_id": F19_POLICY,
        "median_full_rel": median(rels),
        "mean_full_rel": mean(rels),
        "improved_seq_count": sum(1 for v in rels if math.isfinite(v) and v > 0.0),
        "max_harm": max_harm(rels),
        "safety_candidate_pass": safety.get("safety_candidate_pass", False),
        "metric_complete": safety.get("metric_complete", False),
        "all_action_fidelity": safety.get("all_action_fidelity", False),
        "control_taxonomy": controls.get("taxonomy", ""),
        "control_blocker": controls.get("blocker", ""),
        "control_supports_semantic_causality": controls.get("f19_keyframe_control_supports_f19_causality", False),
        "strongest_control_median_full_rel_improvement": controls.get("strongest_control_median_full_rel_improvement"),
    }
    return rows, aggregate


def hook_audit_rows() -> list[dict[str, Any]]:
    checks = [
        ("wrapper_force_non_keyframe", ROOT / "third_party/lingbot-map/benchmark/methods/lingbot_map.py", "force_non_keyframe"),
        ("wrapper_anchor_special_only", ROOT / "third_party/lingbot-map/benchmark/methods/lingbot_map.py", "anchor_special_only"),
        ("wrapper_stage4_policy_env", ROOT / "third_party/lingbot-map/benchmark/methods/lingbot_map.py", "ACL2_V108_STAGE4_POLICY_ID"),
        ("aggregator_special_token_offsets", ROOT / "third_party/lingbot-map/lingbot_map/aggregator/stream.py", "patch_start_idx = 1 + self.num_register_tokens + 1"),
        ("aggregator_kv_cache_skip_append", ROOT / "third_party/lingbot-map/lingbot_map/aggregator/stream.py", "_skip_append"),
        ("window_kv_cache_camera_only", ROOT / "third_party/lingbot-map/lingbot_map/models/gct_stream_window.py", "kv_cache_camera_only"),
        ("attention_layer", ROOT / "third_party/lingbot-map/lingbot_map/layers/attention.py", "attention"),
    ]
    rows: list[dict[str, Any]] = []
    for check_id, path, needle in checks:
        text = path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
        rows.append(
            {
                "schema": "acl2_v111tf_stage0_hook_audit_row_v1",
                "check_id": check_id,
                "path": rel(path),
                "needle": needle,
                "exists": path.exists(),
                "needle_found": needle in text,
                "audit_scope": "readability_only_not_noop_parity",
            }
        )
    return rows


def allowed_memory_action_surfaces() -> list[dict[str, Any]]:
    feasibility = {
        row.get("surface_id", ""): row
        for row in read_csv(V108_STAGE1 / "action_surface_implementation_feasibility.csv")
    }
    return [
        {
            "schema": "acl2_v111tf_stage0_allowed_memory_action_surface_row_v1",
            "memory_family": "Trajectory Memory",
            "v111_track": "Batch T",
            "candidate_surface": "B1 high-risk no-append / trajectory admission",
            "current_status": "existing_hook_verified_in_v110_geometry_but_semantic_controls_pending",
            "required_before_claim": "T1 stronger controls; action fidelity; full 00/01/02/05 metrics",
            "source": rel(V110_FINAL / "final_decision.json"),
        },
        {
            "schema": "acl2_v111tf_stage0_allowed_memory_action_surface_row_v1",
            "memory_family": "Trajectory Memory",
            "v111_track": "Batch T",
            "candidate_surface": "context-token type ablation / soft weighting",
            "current_status": "minimal_hook_audit_required",
            "required_before_claim": "default-off hook; no-action parity; token-type index coverage",
            "source": "v111_plan_stage1_t2",
        },
        {
            "schema": "acl2_v111tf_stage0_allowed_memory_action_surface_row_v1",
            "memory_family": "Anchor Context",
            "v111_track": "Batch A",
            "candidate_surface": "delayed anchor-frame selection / anchor source attention bias",
            "current_status": "hook_feasibility_audit_required",
            "required_before_claim": "anchor source index coverage; no-action parity; delayed latency recording",
            "source": "v111_plan_batch_a",
        },
        {
            "schema": "acl2_v111tf_stage0_allowed_memory_action_surface_row_v1",
            "memory_family": "Local Pose-Reference Window",
            "v111_track": "Batch L",
            "candidate_surface": "local source-token attention bias / query-type-specific read",
            "current_status": "hook_feasibility_audit_required",
            "required_before_claim": "local source index coverage; query type audit; no-action parity",
            "source": "v111_plan_batch_l",
        },
        {
            "schema": "acl2_v111tf_stage0_allowed_memory_action_surface_row_v1",
            "memory_family": "Legacy v108 surfaces",
            "v111_track": "reference",
            "candidate_surface": "A/B/E/F wrapper knobs",
            "current_status": "A/B/E/F were runnable in v108/v110; C/D hook-needed",
            "required_before_claim": "do not reuse as semantic-aware without v111 controls",
            "source": rel(V108_STAGE1 / "action_surface_implementation_feasibility.csv"),
            "v108_A_status": feasibility.get("A", {}).get("implementation_status", ""),
            "v108_B_status": feasibility.get("B", {}).get("implementation_status", ""),
            "v108_C_status": feasibility.get("C", {}).get("implementation_status", ""),
            "v108_D_status": feasibility.get("D", {}).get("implementation_status", ""),
            "v108_E_status": feasibility.get("E", {}).get("implementation_status", ""),
            "v108_F_status": feasibility.get("F", {}).get("implementation_status", ""),
        },
    ]


def forbidden_repeat_text() -> str:
    return """# ACL2 v111TF Forbidden Repeat List

1. Do not claim semantic-aware success from B1 geometry alone. v110R already proved B1 geometry, but semantic causality failed.
2. Do not use debug96, selected windows, L3 proxy, trace movement, or local-only metrics as a full KITTI substitute.
3. Do not repeat B1 threshold sweeps without stronger controls: same-bucket random, schedule-only matched, role rotation, low-risk reverse, and semantic shuffle.
4. Do not treat F19 as semantic-aware. It is a safe internal/keyframe schedule baseline with unresolved semantic causality.
5. Do not mix Anchor Context, Local Window, and Trajectory Memory actions without separate action fidelity and affected-token/frame counts.
6. Do not claim token-type ablation unless camera/register/anchor token indices are separately audited.
7. Do not claim Anchor delayed initialization if only trajectory keyframe selection changed.
8. Do not claim Local query-specific action if the implementation only supports all-query action.
9. Do not use GT, external depth, MoGe, SLAM, Sim(3), pose graph, or trajectory post-processing as runtime cues.
"""


def semantic_controls_text() -> str:
    return """# ACL2 v111TF Semantic Causality Required Controls

For B1/T1:

```text
B1_semantic_only
B1_semantic_plus_internal
B1_internal_only
B1_semantic_shuffle_seed0..9
B1_role_rotation_dynamic_to_stable
B1_role_rotation_dynamic_to_weak
B1_same_count_random_seed0..50
B1_same_bucket_random_seed0..50
B1_schedule_only_matched_seed0..20
B1_low_risk_reverse
B1_high_risk_without_semantic_trust
B1_dynamic_only
B1_boundary_only
B1_weak_context_only
```

Minimum semantic-aware claim boundary:

```text
candidate improvement > same_bucket_random_P95
candidate improvement > same_count_random_P95
candidate improvement > semantic_shuffle_best
candidate improvement > role_rotation_best
candidate improvement > schedule_only_matched_best
low_risk_reverse does not match candidate
semantic+internal >= internal_only + 3 percentage points median improvement
    or semantic-only clearly beats semantic-shuffle / role-rotation controls
```

If these controls match the candidate, taxonomy must be:

```text
FULL_ATE_BOOST_INTERNAL_OR_SCHEDULE_BASELINE_ONLY
```
"""


def known_facts(
    baseline_exact: dict[str, Any],
    b1_aggregate: dict[str, Any],
    f19_aggregate: dict[str, Any],
    hook_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    v108_stage2 = read_json(V108_STAGE2 / "stage2_summary.json")
    v110_final = read_json(V110_FINAL / "final_decision.json")
    return {
        "schema": "acl2_v111tf_stage0_known_facts_v1",
        "baseline_exact_match_by_seq": baseline_exact,
        "semantic_cue_bank": {
            "source": rel(V108_STAGE2 / "stage2_summary.json"),
            "stage2_pass": v108_stage2.get("stage2_pass"),
            "expected_frame_count": v108_stage2.get("expected_frame_count"),
            "processed_frame_count": v108_stage2.get("processed_frame_count"),
            "frame_semantic_coverage": v108_stage2.get("frame_semantic_coverage"),
            "token_semantic_row_count": v108_stage2.get("token_semantic_row_count"),
            "operation_row_count": v108_stage2.get("operation_row_count"),
            "operation_rows_join_coverage_mean": v108_stage2.get("operation_rows_join_coverage_mean"),
        },
        "b1_champion": b1_aggregate,
        "f19_reference": f19_aggregate,
        "v110_final_boundary": {
            "source": rel(V110_FINAL / "final_decision.json"),
            "plan_completed": v110_final.get("plan_completed"),
            "final_taxonomy": v110_final.get("final_taxonomy"),
            "scientific_goal_achieved_as_semantic_aware_method": v110_final.get("scientific_goal_achieved_as_semantic_aware_method"),
            "semantic_causality_pass": v110_final.get("semantic_causality_pass"),
            "full_geometry_pass": v110_final.get("full_geometry_pass"),
        },
        "hook_readability": {
            row["check_id"]: row["needle_found"] for row in hook_rows
        },
    }


def build() -> dict[str, Any]:
    OUT.mkdir(parents=True, exist_ok=True)
    manifest_rows = artifact_manifest()
    baseline_rows, baseline_exact = baseline_table()
    b1_metric_rows, b1_aggregate = b1_rows()
    f19_metric_rows, f19_aggregate = f19_rows()
    hook_rows = hook_audit_rows()
    allowed_rows = allowed_memory_action_surfaces()
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
    b1_metrics_readable = (
        len(b1_metric_rows) == 8
        and b1_aggregate.get("primary_policy_id") in B1_POLICIES
        and math.isfinite(fnum(b1_aggregate.get("median_full_rel")))
        and b1_aggregate.get("final_taxonomy") == "FULL_ATE_BOOST_INTERNAL_SCHEDULE_BASELINE_ONLY"
    )
    f19_metrics_readable = (
        len(f19_metric_rows) == 4
        and f19_aggregate.get("safety_candidate_pass") is True
        and f19_aggregate.get("metric_complete") is True
        and f19_aggregate.get("all_action_fidelity") is True
    )
    semantic_cue_bank = read_json(V108_STAGE2 / "stage2_summary.json")
    semantic_cue_readable = (
        bool(semantic_cue_bank)
        and fnum(semantic_cue_bank.get("frame_semantic_coverage")) >= 0.99
        and int(semantic_cue_bank.get("token_semantic_row_count", 0)) > 0
    )
    hooks_readable = all(bool(row["exists"]) and bool(row["needle_found"]) for row in hook_rows)
    no_stale_lingbot_worker = not pending
    stage0_pass = (
        baseline_exact_match
        and b1_metrics_readable
        and f19_metrics_readable
        and semantic_cue_readable
        and hooks_readable
        and no_stale_lingbot_worker
        and not required_missing
    )
    blockers: list[str] = []
    if not baseline_exact_match:
        blockers.append("baseline_exact_match_failed")
    if not b1_metrics_readable:
        blockers.append("b1_metrics_not_readable_or_boundary_mismatch")
    if not f19_metrics_readable:
        blockers.append("f19_metrics_not_readable")
    if not semantic_cue_readable:
        blockers.append("semantic_cue_bank_not_readable")
    if not hooks_readable:
        blockers.append("lingbot_action_hooks_not_readable")
    if not no_stale_lingbot_worker:
        blockers.append("stale_lingbot_worker_running")
    blockers.extend(f"missing_required_artifact:{item}" for item in required_missing)

    write_csv(OUT / "available_artifact_manifest.csv", manifest_rows)
    write_csv(OUT / "full_kitti_baseline_table.csv", baseline_rows)
    write_csv(OUT / "b1_champion_metric_rows.csv", b1_metric_rows)
    write_csv(OUT / "f19_reference_metric_rows.csv", f19_metric_rows)
    write_csv(OUT / "lingbot_action_hook_audit_rows.csv", hook_rows)
    write_csv(OUT / "allowed_memory_action_surfaces.csv", allowed_rows)
    write_text(OUT / "forbidden_repeat_list.md", forbidden_repeat_text())
    write_text(OUT / "semantic_causality_required_controls.md", semantic_controls_text())
    facts = known_facts(baseline_exact, b1_aggregate, f19_aggregate, hook_rows)
    write_json(OUT / "stage0_known_facts.json", facts)

    summary = {
        "schema": "acl2_v111tf_stage0_evidence_freeze_summary_v1",
        "stage0_pass": stage0_pass,
        "blockers": blockers,
        "baseline_exact_match": baseline_exact_match,
        "baseline_exact_match_by_seq": baseline_exact,
        "b1_metrics_readable": b1_metrics_readable,
        "b1_champion_aggregate": b1_aggregate,
        "f19_metrics_readable": f19_metrics_readable,
        "f19_reference_aggregate": f19_aggregate,
        "semantic_cue_bank_readable": semantic_cue_readable,
        "lingbot_action_hooks_readable": hooks_readable,
        "hook_readability": {row["check_id"]: row["needle_found"] for row in hook_rows},
        "no_stale_lingbot_worker": no_stale_lingbot_worker,
        "pending_lingbot_process_rows": pending,
        "required_missing_artifacts": required_missing,
        "stage0_boundary": {
            "B1": "strong full-ATE internal/schedule baseline, not semantic-aware method",
            "F19": "safe internal/keyframe schedule reference, not semantic-aware method",
            "C_D_or_token_type_hooks": "next hook-audit direction before runtime claim",
            "Anchor_Local_Trajectory": "separate v111 memory families requiring separate fidelity",
        },
        "outputs": {
            "stage0_summary": rel(OUT / "stage0_summary.json"),
            "stage0_known_facts": rel(OUT / "stage0_known_facts.json"),
            "full_kitti_baseline_table": rel(OUT / "full_kitti_baseline_table.csv"),
            "b1_champion_metric_rows": rel(OUT / "b1_champion_metric_rows.csv"),
            "f19_reference_metric_rows": rel(OUT / "f19_reference_metric_rows.csv"),
            "allowed_memory_action_surfaces": rel(OUT / "allowed_memory_action_surfaces.csv"),
            "forbidden_repeat_list": rel(OUT / "forbidden_repeat_list.md"),
            "semantic_causality_required_controls": rel(OUT / "semantic_causality_required_controls.md"),
            "lingbot_action_hook_audit_rows": rel(OUT / "lingbot_action_hook_audit_rows.csv"),
            "available_artifact_manifest": rel(OUT / "available_artifact_manifest.csv"),
        },
    }
    write_json(OUT / "stage0_summary.json", summary)
    if not stage0_pass:
        write_text(
            OUT / "STAGE0_EVIDENCE_FREEZE_BLOCKED.md",
            "# ACL2 v111TF Stage0 Evidence Freeze Blocked\n\n"
            f"blockers: `{blockers}`\n\n"
            f"pending_lingbot_process_rows: `{pending}`\n",
        )
    return summary


def main() -> None:
    print(json.dumps(clean_json(build()), indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
