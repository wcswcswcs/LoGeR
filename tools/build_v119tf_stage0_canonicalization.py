#!/usr/bin/env python3
"""Build ACL2 v119-TF Stage0 canonicalization artifacts.

This builder is deliberately conservative: it freezes current code/config/
checkpoint hashes and imports prior v118 reference evidence, but it does not
claim a v119 fresh-baseline pass unless the required evidence is present.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "results/acl2_v119tf_semantic_addressable_geometry_carrier_routing_representation_repair"
STAGE0 = RESULT_ROOT / "stage0"
V118_ROOT = ROOT / "results/acl2_v118tf_operation_specific_semantic_carrier_calibration"
V118_STAGE0 = V118_ROOT / "stage0_fresh_reference"

PLAN = ROOT / "docs/ACL2_v119TF_SemanticAddressableGeometryCarrierRouting_RepresentationRepair_ExhaustiveExperimentPlan.md"
EXEC_LOG = ROOT / "docs/ACL2_v119TF_SemanticAddressableGeometryCarrierRouting_RepresentationRepair_执行日志.md"
RECAP_LOG = ROOT / "docs/ACL2_v119TF_SemanticAddressableGeometryCarrierRouting_RepresentationRepair_实验结果复盘.md"

V118_CLOSURE = V118_ROOT / "V118_POSTFINAL_R82_CURRENT_CLOSURE_SUMMARY.json"
V118_FINAL = V118_ROOT / "V118_FINAL_DECISION_SUMMARY.json"
V118_MATRIX = V118_ROOT / "V118_CURRENT_BRANCH_COMPLETION_MATRIX.csv"
V118_REGISTRY = V118_ROOT / "V118_RUN_REGISTRY.csv"
V118_BASELINES = V118_STAGE0 / "stage0_baseline_metric_rows.csv"
V118_B1 = V118_STAGE0 / "stage0_b1_reference_rows.csv"
V118_HS_GENERIC = V118_STAGE0 / "stage0_hs_generic_carrier_rows.csv"
V119_HS_FRESH = RESULT_ROOT / "stage0_hs_fresh_baselines"
V119_HS_ROWMEAN_FRESH = RESULT_ROOT / "stage0_hs_rowmean_mrt_tight_fresh"
V119_LB_FRESH = RESULT_ROOT / "stage0_lingbot_fresh_baselines"
V119_LB_B1_FRESH = RESULT_ROOT / "stage0_lingbot_b1_fresh_baselines"

CODE_PATHS = [
    "tools/build_v119tf_stage0_canonicalization.py",
    "tools/build_v119tf_core_code_audit_pack.py",
    "tools/build_v119tf_lingbot_stage0_fresh_baseline_configs.py",
    "tools/build_v119tf_lingbot_stage0_fresh_metrics.py",
    "tools/build_v119tf_lingbot_b1_stage0_fresh_configs.py",
    "tools/build_v119tf_lingbot_b1_stage0_fresh_metrics.py",
    "tools/build_v118tf_stage0_fresh_reference.py",
    "tools/build_v118tf_stage1_causal_object_track_sidecar.py",
    "tools/build_v118tf_stage2_memory_entry_provenance.py",
    "tools/build_v118tf_stage3_internal_signal_readiness.py",
    "tools/build_v118tf_stage4_r79_lingbot_ar_g1p25_source_subset_audit.py",
    "tools/build_v118tf_stage4_r80_lingbot_ar_s125_gain_control_matrix.py",
    "tools/build_v118tf_stage4_r81_lingbot_ar_s125_query_bias_matrix.py",
    "tools/build_v118tf_stage4_r82_current_closure_after_lbar_r81.py",
    "tools/run_v118tf_policy_manifest_batch.py",
    "tools/run_v118tf_hs_gla_pilot_matrix.py",
    "tools/run_v118tf_hs_gla_oom_repair_matrix.py",
    "third_party/lingbot-map/benchmark/methods/lingbot_map.py",
    "third_party/lingbot-map/lingbot_map/aggregator/stream.py",
    "third_party/lingbot-map/lingbot_map/layers/attention.py",
    "third_party/lingbot-map/lingbot_map/layers/flashinfer_cache.py",
    "third_party/lingbot-map/lingbot_map/models/gct_stream.py",
    "third_party/lingbot-map/lingbot_map/models/gct_stream_window.py",
    "third_party/HorizonStream/horizonstream/runtime/layers/attention.py",
    "third_party/HorizonStream/horizonstream/runtime/models/horizonstream.py",
    "third_party/HorizonStream/horizonstream/runtime/semantic_runtime.py",
    "third_party/HorizonStream/horizonstream/models/horizonstream.py",
    "eval/relpose/evo_utils.py",
    "run_pipeline_abc.py",
    "run_pipeline_abc_v2.py",
]

CONFIG_PATHS = [
    "third_party/lingbot-map/benchmark/configs/kitti.yaml",
    "third_party/lingbot-map/benchmark/configs/datasets/kitti_504x280.yaml",
    "third_party/lingbot-map/benchmark/configs/methods/lingbot_map.yaml",
    "third_party/lingbot-map/benchmark/configs/methods/lingbot_map_v1.yaml",
    "third_party/HorizonStream/configs/horizonstream_infer.yaml",
    "results/acl2_v105tf_dualtrack_lingbot_loger_evidence_eligibility/configs/kitti_lingbot_stream_default.yaml",
    "results/acl2_v105tf_dualtrack_lingbot_loger_evidence_eligibility/configs/methods/lingbot_map_stream_default.yaml",
]

CHECKPOINTS = [
    ("lingbot_map_long", "third_party/lingbot-map/checkpoints/lingbot-map-long.pt"),
    ("horizonstream", "third_party/HorizonStream/checkpoints/HorizonStream.pt"),
]

REQUIRED_BASELINES = [
    ("LingBot", "lingbot_map_stream_default_flashinfer", "00"),
    ("LingBot", "lingbot_map_stream_default_flashinfer", "01"),
    ("LingBot", "lingbot_map_stream_default_flashinfer", "02"),
    ("LingBot", "lingbot_map_stream_default_flashinfer", "05"),
    ("HorizonStream", "horizonstream_no_loop_noaction", "00"),
    ("HorizonStream", "horizonstream_no_loop_noaction", "01"),
    ("HorizonStream", "horizonstream_no_loop_noaction", "02"),
    ("HorizonStream", "horizonstream_no_loop_noaction", "05"),
    ("LingBot", "B1_strong_carrier_reference", "00"),
    ("LingBot", "B1_strong_carrier_reference", "02"),
    ("HorizonStream", "rowmean_mrt_tight_generic_carrier", "00"),
    ("HorizonStream", "rowmean_mrt_tight_generic_carrier", "02"),
]

V119_BRANCHES = [
    ("LB-AI-FIX", "LingBot", "Anchor initialization representation repair", "LB-Anchor", "LB-SCHED+SEM-V3"),
    ("LB-AR-FIX", "LingBot", "Anchor read representation repair", "LB-Anchor", "LB-NORM+SEM-V3+source/query span"),
    ("LB-LR", "LingBot", "Local read", "LB-Local", "LB-NORM+SEM-V3+source/query span"),
    ("LB-TA", "LingBot", "Trajectory admission", "LB-Trajectory", "LB-LOGICAL+SEM-V3"),
    ("LB-TR", "LingBot", "Trajectory retrieval", "LB-Trajectory", "LB-LOGICAL+SEM-V3"),
    ("LB-TE", "LingBot", "Retention / eviction", "LB-Trajectory", "LB-LOGICAL+SEM-V3"),
    ("LB-CT", "LingBot", "Compact context token routing", "LB-Local", "LB-LOGICAL+SEM-V3"),
    ("HS-PW", "HorizonStream", "Persistent write", "HS-GLA", "HS-KDA selected-layer instrumentation"),
    ("HS-GR", "HorizonStream", "GLA retention", "HS-GLA", "direct gamma/decay hook or three-repair blocker"),
    ("HS-RR", "HorizonStream", "Readout routing", "HS-Readout", "persistent/transient lane trace"),
]


def rel(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def clean_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clean_json(item) for item in value]
    if isinstance(value, Path):
        return rel(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean_json(data), indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = []
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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_state() -> dict[str, Any]:
    rev = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True, check=False)
    status = subprocess.run(["git", "status", "--short"], cwd=ROOT, text=True, capture_output=True, check=False)
    return {
        "git_head": rev.stdout.strip() if rev.returncode == 0 else "",
        "git_head_returncode": rev.returncode,
        "git_status_short_returncode": status.returncode,
        "git_status_short_line_count": len([line for line in status.stdout.splitlines() if line.strip()]),
        "git_status_short_sample": status.stdout.splitlines()[:120],
    }


def hash_file_row(artifact_id: str, rel_path: str, category: str) -> dict[str, Any]:
    path = ROOT / rel_path
    row: dict[str, Any] = {
        "artifact_id": artifact_id,
        "category": category,
        "path": rel_path,
        "exists": path.is_file(),
        "size_bytes": path.stat().st_size if path.is_file() else "",
        "mtime_ns": path.stat().st_mtime_ns if path.is_file() else "",
        "sha256": "",
        "hash_status": "missing",
    }
    if path.is_file():
        row["sha256"] = sha256(path)
        row["hash_status"] = "current_turn_recomputed"
    return row


def current_code_hashes(generated_at: str) -> dict[str, Any]:
    rows = [hash_file_row(Path(item).stem, item, "code") for item in CODE_PATHS]
    combined = hashlib.sha256()
    for row in sorted(rows, key=lambda item: item["path"]):
        combined.update(str(row.get("path", "")).encode())
        combined.update(str(row.get("sha256", "")).encode())
    return {
        "schema": "acl2_v119tf_stage0_current_code_hashes_v1",
        "generated_at_utc": generated_at,
        "git_state": git_state(),
        "combined_code_hash": combined.hexdigest(),
        "hash_rows": rows,
        "missing_count": sum(1 for row in rows if not row["exists"]),
    }


def current_config_hashes(generated_at: str) -> dict[str, Any]:
    rows = [hash_file_row(Path(item).stem, item, "config") for item in CONFIG_PATHS]
    combined = hashlib.sha256()
    for row in sorted(rows, key=lambda item: item["path"]):
        combined.update(str(row.get("path", "")).encode())
        combined.update(str(row.get("sha256", "")).encode())
    return {
        "schema": "acl2_v119tf_stage0_current_config_hashes_v1",
        "generated_at_utc": generated_at,
        "combined_config_hash": combined.hexdigest(),
        "hash_rows": rows,
        "missing_count": sum(1 for row in rows if not row["exists"]),
    }


def current_checkpoint_hashes(generated_at: str) -> dict[str, Any]:
    rows = []
    for artifact_id, rel_path in CHECKPOINTS:
        path = ROOT / rel_path
        print(f"[stage0] hashing checkpoint {rel_path}", flush=True)
        rows.append(hash_file_row(artifact_id, rel_path, "checkpoint"))
    combined = hashlib.sha256()
    for row in sorted(rows, key=lambda item: item["path"]):
        combined.update(str(row.get("path", "")).encode())
        combined.update(str(row.get("sha256", "")).encode())
    return {
        "schema": "acl2_v119tf_stage0_current_checkpoint_hashes_v1",
        "generated_at_utc": generated_at,
        "combined_checkpoint_hash": combined.hexdigest(),
        "hash_rows": rows,
        "missing_count": sum(1 for row in rows if not row["exists"]),
    }


def source_hash_lookup(*hash_docs: dict[str, Any]) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for doc in hash_docs:
        for row in doc.get("hash_rows", []):
            lookup[str(row.get("path", ""))] = str(row.get("sha256", ""))
    return lookup


def canonical_run_id(model: str, branch: str, policy: str, seq: str, code_hash: str, config_hash: str, seed: str, timestamp: str) -> str:
    payload = f"{model}/{branch}/{policy}/{seq}/{code_hash[:16]}/{config_hash[:16]}/{seed}/{timestamp}"
    return hashlib.sha256(payload.encode()).hexdigest()[:24]


def first_row(rows: list[dict[str, str]], pred) -> dict[str, str]:
    for row in rows:
        if pred(row):
            return row
    return {}


def fresh_hs_noaction_row(seq: str) -> dict[str, Any]:
    output_root = V119_HS_FRESH / "outputs" / f"v119_stage0_hs_noaction_baseline_full_kitti_{seq}"
    summary_path = output_root / "pipeline_summary.json"
    eval_path = output_root / "eval_summary.json"
    manifest_path = V119_HS_FRESH / "diagnostics" / f"v119_stage0_hs_noaction_baseline_full_kitti_{seq}" / "run_manifest.json"
    summary = read_json(summary_path)
    manifest = read_json(manifest_path)
    key = f"{seq}/02"
    seq_row = (summary.get("infer_eval", {}).get("sequences", {}) or {}).get(key, {})
    metric = (seq_row.get("metrics", {}) or {}).get("main", {})
    if manifest.get("returncode") != 0 or not summary.get("ran_eval") or not metric:
        return {}
    return {
        "source_path": rel(summary_path),
        "eval_path": rel(eval_path),
        "manifest_path": rel(manifest_path),
        "full_ATE_sim3": metric.get("ate_rmse", ""),
        "num_pose_pairs": metric.get("num_pose_pairs", ""),
        "global_sim3_scale": metric.get("sim3_scale", ""),
        "ate_mean": metric.get("ate_mean", ""),
        "ate_median": metric.get("ate_median", ""),
        "returncode": manifest.get("returncode", ""),
        "command": manifest.get("command", ""),
    }


def fresh_hs_rowmean_mrt_tight_row(seq: str) -> dict[str, Any]:
    case = f"v119_stage0_hs_rowmean_mrt_scaledelta_tight_generic_carrier_full_kitti_{seq}"
    output_root = V119_HS_ROWMEAN_FRESH / "outputs" / case
    summary_path = output_root / "pipeline_summary.json"
    eval_path = output_root / "eval_summary.json"
    manifest_path = V119_HS_ROWMEAN_FRESH / "diagnostics" / case / "run_manifest.json"
    summary = read_json(summary_path)
    manifest = read_json(manifest_path)
    key = f"{seq}/02"
    seq_row = (summary.get("infer_eval", {}).get("sequences", {}) or {}).get(key, {})
    metric = (seq_row.get("metrics", {}) or {}).get("main", {})
    baseline = fresh_hs_noaction_row(seq)
    if manifest.get("returncode") != 0 or not summary.get("ran_eval") or not metric or not baseline:
        return {}
    baseline_ate = float(baseline["full_ATE_sim3"])
    candidate_ate = float(metric["ate_rmse"])
    return {
        "source_path": rel(summary_path),
        "eval_path": rel(eval_path),
        "manifest_path": rel(manifest_path),
        "full_ATE_sim3": candidate_ate,
        "baseline_full_ATE_sim3": baseline_ate,
        "full_ATE_rel_improvement": (baseline_ate - candidate_ate) / baseline_ate,
        "num_pose_pairs": metric.get("num_pose_pairs", ""),
        "global_sim3_scale": metric.get("sim3_scale", ""),
        "ate_mean": metric.get("ate_mean", ""),
        "ate_median": metric.get("ate_median", ""),
        "returncode": manifest.get("returncode", ""),
        "command": manifest.get("command", ""),
    }


def fresh_lingbot_default_row(seq: str) -> dict[str, Any]:
    rows = read_csv(V119_LB_FRESH / "full_sequence_metrics/stage0_lingbot_flashinfer_baseline_rows.csv")
    metric = first_row(rows, lambda row: row.get("seq") == seq and row.get("reference_source") == "v119_fresh_flashinfer_baseline_rerun")
    if not metric:
        return {}
    method_root = ROOT / metric.get("method_root", "")
    complete = method_root / ".complete.json"
    traj = method_root / "traj.txt"
    eval_traj = method_root / "eval/traj.json"
    if not (complete.is_file() and traj.is_file() and eval_traj.is_file()):
        return {}
    metric["source_path"] = metric.get(
        "source_path",
        rel(V119_LB_FRESH / "full_sequence_metrics/stage0_lingbot_flashinfer_baseline_rows.csv"),
    )
    metric["command"] = "see stage0_lingbot_fresh_baselines/run_manifest.csv"
    metric["num_pose_pairs"] = metric.get("num_frames", "")
    return metric


def fresh_lingbot_b1_row(seq: str) -> dict[str, Any]:
    rows = read_csv(V119_LB_B1_FRESH / "full_sequence_metrics/stage0_lingbot_b1_fresh_rows.csv")
    metric = first_row(
        rows,
        lambda row: row.get("seq") == seq
        and row.get("reference_source") == "v119_fresh_B1_semantic_only_flashinfer_rerun"
        and row.get("action_fidelity_pass") in {"True", "true", "1", "True"},
    )
    if not metric:
        return {}
    method_root = ROOT / metric.get("method_root", "")
    complete = method_root / ".complete.json"
    traj = method_root / "traj.txt"
    eval_traj = method_root / "eval/traj.json"
    action_file = ROOT / metric.get("action_file", "")
    if not (complete.is_file() and traj.is_file() and eval_traj.is_file() and action_file.is_file()):
        return {}
    metric["source_path"] = metric.get(
        "source_path",
        rel(V119_LB_B1_FRESH / "full_sequence_metrics/stage0_lingbot_b1_fresh_rows.csv"),
    )
    metric["command"] = "see stage0_lingbot_b1_fresh_baselines/run_manifest.csv"
    metric["num_pose_pairs"] = metric.get("num_frames", "")
    return metric


def build_baseline_rows(generated_at: str, hashes: dict[str, str], code_hash: str, config_hash: str) -> list[dict[str, Any]]:
    v118_base = read_csv(V118_BASELINES)
    v118_b1 = read_csv(V118_B1)
    v118_hs = read_csv(V118_HS_GENERIC)
    rows: list[dict[str, Any]] = []
    for model, policy, seq in REQUIRED_BASELINES:
        branch = "Stage0"
        source_path = ""
        status = "missing_required_v119_fresh_or_strict_baseline"
        verification = "missing"
        metric: dict[str, str] = {}
        if model == "LingBot" and policy == "lingbot_map_stream_default_flashinfer":
            fresh_metric = fresh_lingbot_default_row(seq)
            if fresh_metric:
                metric = fresh_metric
                source_path = fresh_metric.get("source_path", "")
                status = "v119_fresh_baseline_rerun_pass"
                verification = "fresh_v119_lingbot_flashinfer_run_returncode0_eval_traj_present"
            else:
                metric = first_row(v118_base, lambda row: row.get("model") == "LingBot" and row.get("seq") == seq)
                source_path = metric.get("source_path", "")
                if metric:
                    status = "reference_imported_not_v119_fresh"
                    verification = "v118_stage0_reference_available_current_hashes_captured_v119_rerun_pending"
        elif model == "HorizonStream" and policy == "horizonstream_no_loop_noaction":
            fresh_metric = fresh_hs_noaction_row(seq)
            if fresh_metric:
                metric = fresh_metric
                source_path = fresh_metric.get("source_path", "")
                status = "v119_fresh_baseline_rerun_pass"
                verification = "fresh_v119_run_returncode0_pipeline_summary_present"
            else:
                metric = first_row(v118_base, lambda row: row.get("model") == "HorizonStream" and row.get("seq") == seq)
                source_path = metric.get("source_path", "")
                if metric:
                    status = "reference_imported_not_v119_fresh"
                    verification = "v118_stage0_reference_available_for_00_02_only_current_hashes_captured_v119_0105_missing"
        elif model == "LingBot" and policy == "B1_strong_carrier_reference":
            fresh_metric = fresh_lingbot_b1_row(seq)
            if fresh_metric:
                metric = fresh_metric
                source_path = fresh_metric.get("source_path", "")
                status = "v119_fresh_baseline_rerun_pass"
                verification = "fresh_v119_lingbot_b1_flashinfer_replacement_returncode0_eval_traj_action_fidelity_pass"
            else:
                metric = first_row(
                    v118_b1,
                    lambda row: row.get("source_id") in {"v110_stage4_B1_reference", "v116_task1_AB0_B1_reference"} and row.get("seq") == seq,
                )
                source_path = metric.get("source_path", "")
                if metric:
                    status = "strong_reference_imported_not_v119_fresh"
                    verification = "v118_stage0_strong_carrier_reference_available_control_boundary_preserved"
        elif model == "HorizonStream" and policy == "rowmean_mrt_tight_generic_carrier":
            fresh_metric = fresh_hs_rowmean_mrt_tight_row(seq)
            if fresh_metric:
                metric = fresh_metric
                source_path = fresh_metric.get("source_path", "")
                status = "v119_fresh_baseline_rerun_pass"
                verification = "fresh_v119_rowmean_mrt_tight_run_returncode0_vs_fresh_noaction_baseline"
            else:
                metric = first_row(v118_hs, lambda row: row.get("source_id") == "v116_rowmean_mrt_tight_vs_current_noaction" and row.get("seq") == seq)
                source_path = metric.get("source_path", "")
                if metric:
                    status = "generic_carrier_reference_imported_not_v119_fresh"
                    verification = "v118_stage0_generic_carrier_reference_available_control_boundary_preserved"
        run_id = canonical_run_id(model, branch, policy, seq, code_hash, config_hash, "0", generated_at)
        rows.append(
            {
                "schema": "acl2_v119tf_stage0_canonical_baseline_row_v1",
                "run_id": run_id,
                "model": model,
                "branch": branch,
                "policy_id": policy,
                "seq": seq,
                "code_hash": code_hash,
                "config_hash": config_hash,
                "seed": "0",
                "timestamp_utc": generated_at,
                "status": status,
                "strict_verification_status": verification,
                "full_ATE_sim3": metric.get("full_ATE_sim3", metric.get("candidate_full_ATE_sim3_rmse", "")),
                "baseline_full_ATE_sim3": metric.get("baseline_full_ATE_sim3", metric.get("baseline_full_ATE_sim3_rmse", "")),
                "full_ATE_rel_improvement": metric.get("full_ATE_rel_improvement", ""),
                "rolling_p90": metric.get("rolling_ATE_p90", ""),
                "rolling_p90_rel_improvement": metric.get("rolling_p90_rel_improvement", ""),
                "segment_scale_log_error_median_abs": metric.get("segment_scale_log_error_median_abs", ""),
                "action_fidelity_pass": metric.get("action_fidelity_pass", ""),
                "source_artifact": source_path,
                "runtime_command": metric.get("command", ""),
                "num_pose_pairs": metric.get("num_pose_pairs", ""),
                "global_sim3_scale": metric.get("global_sim3_scale", ""),
                "claim_boundary": "Stage0 reference row only; not a v119 runtime or semantic success claim",
            }
        )
    return rows


def frozen_negative_boundaries() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    matrix = read_csv(V118_MATRIX)
    for row in matrix:
        rows.append(
            {
                "schema": "acl2_v119tf_stage0_frozen_negative_boundary_row_v1",
                "source_version": "v118",
                "source_branch": row.get("branch", ""),
                "model": row.get("model", ""),
                "operation": row.get("operation", ""),
                "surface": row.get("surface", ""),
                "terminal_status": row.get("status", ""),
                "latest_decision": row.get("latest_decision", ""),
                "global_goal_achieved": row.get("global_goal_achieved", ""),
                "primary_blocker": row.get("primary_blocker", ""),
                "evidence_count": row.get("evidence_count", ""),
                "artifact": row.get("artifact", ""),
                "v119_reuse_boundary": "negative/control boundary only; do not rerun v118 scalar/post-mix route as v119 success",
            }
        )
    return rows


def frozen_strong_carriers() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in read_csv(V118_B1):
        rows.append(
            {
                "schema": "acl2_v119tf_stage0_frozen_strong_carrier_row_v1",
                "source_version": "v118",
                "carrier_family": "LingBot_B1_or_control",
                "source_id": row.get("source_id", ""),
                "policy_id": row.get("policy_id", ""),
                "seq": row.get("seq", ""),
                "full_ATE_rel_improvement": row.get("full_ATE_rel_improvement", ""),
                "action_fidelity_pass": row.get("action_fidelity_pass", ""),
                "source_path": row.get("source_path", ""),
                "claim_boundary": row.get("claim_boundary", "reference/control carrier only"),
            }
        )
    for row in read_csv(V118_HS_GENERIC):
        rows.append(
            {
                "schema": "acl2_v119tf_stage0_frozen_strong_carrier_row_v1",
                "source_version": "v118",
                "carrier_family": "HorizonStream_generic_rowmean_mrt",
                "source_id": row.get("source_id", ""),
                "policy_id": row.get("candidate_variant", ""),
                "seq": row.get("seq", ""),
                "full_ATE_rel_improvement": row.get("full_ATE_rel_improvement", ""),
                "action_fidelity_pass": "",
                "source_path": row.get("source_path", ""),
                "claim_boundary": row.get("claim_boundary", "generic carrier only"),
            }
        )
    return rows


def stale_artifact_audit(generated_at: str) -> list[dict[str, Any]]:
    checks = [
        ("v119_result_root", RESULT_ROOT, "created_by_this_stage0_if_missing"),
        ("v118_r82_closure", V118_CLOSURE, "authoritative_v118_current_boundary"),
        ("v118_r60_closure", V118_ROOT / "V118_POSTFINAL_R60_CURRENT_CLOSURE_SUMMARY.json", "superseded_by_r82"),
        ("v118_r33_closure", V118_ROOT / "V118_POSTFINAL_R33_CURRENT_CLOSURE_SUMMARY.json", "superseded_by_r82"),
        ("v118_final_summary", V118_FINAL, "matches_r82_boundary_if_decision_equal"),
        ("v118_current_matrix", V118_MATRIX, "authoritative_branch_matrix"),
        ("superseded_code_pack_first_try", ROOT / "code_audit_pack/acl2_v119tf_core_code_audit_20260713_223927.zip", "superseded_by_224139_pack"),
        ("final_code_pack", ROOT / "code_audit_pack/acl2_v119tf_core_code_audit_20260713_224139.zip", "current_code_pack_for_review"),
    ]
    rows = []
    for artifact_id, path, status in checks:
        rows.append(
            {
                "schema": "acl2_v119tf_stage0_stale_artifact_audit_row_v1",
                "generated_at_utc": generated_at,
                "artifact_id": artifact_id,
                "path": rel(path),
                "exists": path.exists(),
                "size_bytes": path.stat().st_size if path.exists() and path.is_file() else "",
                "status": status,
            }
        )
    proc = subprocess.run(["ps", "-eo", "pid,ppid,user,stat,etime,cmd"], cwd=ROOT, text=True, capture_output=True, check=False)
    markers = ("v118", "v119", "lingbot", "horizonstream", "run_v118", "run_v119")
    stale = []
    current_pid = str(os.getpid())
    for line in proc.stdout.splitlines()[1:]:
        lower = line.lower()
        parts = line.split(None, 5)
        pid = parts[0] if parts else ""
        if pid == current_pid:
            continue
        if "build_v119tf_stage0_canonicalization.py" in lower:
            continue
        if "build_v119tf_core_code_audit_pack.py" in lower:
            continue
        if any(marker in lower for marker in markers) and "ps -eo" not in lower and "rg " not in lower:
            stale.append(line.strip())
    rows.append(
        {
            "schema": "acl2_v119tf_stage0_stale_artifact_audit_row_v1",
            "generated_at_utc": generated_at,
            "artifact_id": "candidate_stale_processes",
            "path": "",
            "exists": bool(stale),
            "size_bytes": "",
            "status": "none_detected" if not stale else "manual_review_required:" + " | ".join(stale[:5]),
        }
    )
    return rows


def canonical_schema_doc() -> str:
    return """# V119 Canonical Result Schema

Immutable run rows must be keyed by:

```text
model/branch/policy/seq/code_hash/config_hash/seed/timestamp
```

Required canonical fields:

```text
run_id
model
branch
policy_id
seq
code_hash
config_hash
checkpoint_hash
seed
timestamp_utc
runtime_command
source_artifact
status
metric_role
full_ATE_sim3
baseline_full_ATE_sim3
full_ATE_rel_improvement
rolling_p90
rolling_p90_rel_improvement
segment_scale_log_error_median_abs
action_fidelity_pass
control_family
matched_control_status
claim_boundary
```

Rules:

- Final tables are built only from immutable run rows, never by manually
  overwriting old summaries.
- Imported v118 rows must carry `metric_role=stage0_reference_import` and
  cannot be promoted to v119 runtime evidence without fresh rerun or strict
  current-code verification.
- Missing required baseline rows must stay explicit; do not infer metrics from
  a related sequence or policy.
"""


def build_global_metric_rows(baselines: list[dict[str, Any]], checkpoint_hash: str) -> list[dict[str, Any]]:
    rows = []
    for row in baselines:
        status = str(row["status"])
        metric_role = "stage0_fresh_baseline_rerun" if status == "v119_fresh_baseline_rerun_pass" else "stage0_reference_import"
        matched_control_status = (
            "fresh_v119_noaction_baseline"
            if status == "v119_fresh_baseline_rerun_pass"
            else "not_applicable_stage0_reference"
        )
        rows.append(
            {
                "schema": "acl2_v119tf_canonical_metric_row_v1",
                "run_id": row["run_id"],
                "model": row["model"],
                "branch": row["branch"],
                "policy_id": row["policy_id"],
                "seq": row["seq"],
                "code_hash": row["code_hash"],
                "config_hash": row["config_hash"],
                "checkpoint_hash": checkpoint_hash,
                "seed": row["seed"],
                "timestamp_utc": row["timestamp_utc"],
                "metric_role": metric_role,
                "status": status,
                "strict_verification_status": row.get("strict_verification_status", ""),
                "full_ATE_sim3": row["full_ATE_sim3"],
                "baseline_full_ATE_sim3": row["baseline_full_ATE_sim3"],
                "full_ATE_rel_improvement": row["full_ATE_rel_improvement"],
                "rolling_p90": row["rolling_p90"],
                "rolling_p90_rel_improvement": row["rolling_p90_rel_improvement"],
                "segment_scale_log_error_median_abs": row["segment_scale_log_error_median_abs"],
                "action_fidelity_pass": row["action_fidelity_pass"],
                "num_pose_pairs": row.get("num_pose_pairs", ""),
                "global_sim3_scale": row.get("global_sim3_scale", ""),
                "runtime_command": row.get("runtime_command", ""),
                "control_family": "",
                "matched_control_status": matched_control_status,
                "source_artifact": row["source_artifact"],
                "claim_boundary": row["claim_boundary"],
            }
        )
    return rows


def branch_completion_rows(generated_at: str) -> list[dict[str, Any]]:
    return [
        {
            "schema": "acl2_v119tf_branch_completion_row_v1",
            "branch": branch,
            "model": model,
            "operation": operation,
            "surface": surface,
            "mandatory_stage1_dependency": dependency,
            "status": "PENDING_STAGE1_REPRESENTATION_REPAIR",
            "terminal": False,
            "latest_decision": "",
            "missing_required_variant_count": "",
            "unmatched_control_count": "",
            "not_run_count": "",
            "unexplained_pending_count": "",
            "global_goal_achieved": False,
            "updated_at_utc": generated_at,
            "artifact": "",
        }
        for branch, model, operation, surface, dependency in V119_BRANCHES
    ]


def run_registry_rows(generated_at: str, code_hash: str, config_hash: str, checkpoint_hash: str) -> list[dict[str, Any]]:
    return [
        {
            "schema": "acl2_v119tf_run_registry_row_v1",
            "run_id": canonical_run_id("global", "Stage0", "canonicalization", "all", code_hash, config_hash, "0", generated_at),
            "model": "global",
            "branch": "Stage0",
            "policy_id": "canonicalization",
            "seq": "all",
            "code_hash": code_hash,
            "config_hash": config_hash,
            "checkpoint_hash": checkpoint_hash,
            "seed": "0",
            "timestamp_utc": generated_at,
            "command": "python3 tools/build_v119tf_stage0_canonicalization.py",
            "status": "stage0_artifacts_written_gate_incomplete",
            "source_artifact": rel(STAGE0 / "stage0_summary.json"),
        }
    ]


def write_code_audit(generated_at: str) -> None:
    text = f"""# V119 Code Change Audit

Generated at: `{generated_at}`

## Changes Made So Far

| file | change type | runtime behavior changed | purpose |
|---|---|---:|---|
| `tools/build_v119tf_core_code_audit_pack.py` | added | false | Build compact reviewer-facing source/artifact zip under `code_audit_pack`. |
| `tools/build_v119tf_stage0_canonicalization.py` | added | false | Build Stage0 canonicalization/hash/boundary artifacts. |
| `docs/ACL2_v119TF_SemanticAddressableGeometryCarrierRouting_RepresentationRepair_执行日志.md` | appended | false | Record commands, files, and verification evidence. |
| `docs/ACL2_v119TF_SemanticAddressableGeometryCarrierRouting_RepresentationRepair_实验结果复盘.md` | appended | false | Record analysis, blockers, and audit boundaries. |

## Runtime Modifications

No LingBot, HorizonStream, LoGeR, evaluator, or policy runtime code has been
modified yet in v119. Stage1 representation repairs remain pending.
"""
    write_text(RESULT_ROOT / "V119_CODE_CHANGE_AUDIT.md", text)


def write_initial_final_docs(generated_at: str, stage0_pass: bool, blockers: list[str]) -> None:
    summary = {
        "schema": "acl2_v119tf_final_decision_summary_v0_stage0_initialized",
        "created_at_utc": generated_at,
        "global_goal_achieved": False,
        "stage0_gate_pass": stage0_pass,
        "stage0_blockers": blockers,
        "decision": "V119_NOT_COMPLETE_STAGE0_INITIALIZED_ONLY",
        "truthfulness_boundary": "This file is an initialized placeholder after Stage0 canonicalization, not a final experiment decision.",
    }
    write_json(RESULT_ROOT / "V119_FINAL_DECISION_SUMMARY.json", summary)
    report = f"""# V119 Final Decision Report

Generated at: `{generated_at}`

Current decision: `V119_NOT_COMPLETE_STAGE0_INITIALIZED_ONLY`

Stage0 gate pass: `{stage0_pass}`

Blockers:

```text
{chr(10).join(blockers) if blockers else "none"}
```

This is not a final v119 result. Stage1 representation repair, counterfactual
gate, runtime pilot, blind validation, and branch completion matrix are not
complete.
"""
    write_text(RESULT_ROOT / "V119_FINAL_DECISION_REPORT.md", report)
    boundaries = """# V119 Method And No-Go Boundaries

- v118 scalar/risk/stability-only routes remain frozen negative boundaries.
- Imported v118 Stage0 references are baseline/control evidence only.
- No v119 runtime branch may launch before its Stage1 dependencies pass.
- No semantic success is claimed before counterfactual, pilot, blind, and
  matched-control gates pass.
"""
    write_text(RESULT_ROOT / "V119_METHOD_AND_NO_GO_BOUNDARIES.md", boundaries)
    write_csv(
        RESULT_ROOT / "V119_COUNTERFACTUAL_RANKING_SUMMARY.csv",
        [
            {
                "schema": "acl2_v119tf_counterfactual_ranking_summary_row_v1",
                "status": "not_run_stage0_only",
                "spearman": "",
                "auroc": "",
                "topq_uplift_minus_p95_controls": "",
                "direction_00_02_consistent": "",
                "claim_boundary": "counterfactual gate not attempted yet",
            }
        ],
    )


def stage0_summary(
    generated_at: str,
    code_doc: dict[str, Any],
    config_doc: dict[str, Any],
    checkpoint_doc: dict[str, Any],
    baselines: list[dict[str, Any]],
    stale_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    required_missing = [row for row in baselines if str(row["status"]).startswith("missing")]
    imported_not_fresh = [row for row in baselines if "not_v119_fresh" in str(row["status"])]
    stale_process_rows = [row for row in stale_rows if row["artifact_id"] == "candidate_stale_processes" and row["status"] != "none_detected"]
    blockers = []
    if code_doc["missing_count"]:
        blockers.append("current_code_hash_missing_files")
    if config_doc["missing_count"]:
        blockers.append("current_config_hash_missing_files")
    if checkpoint_doc["missing_count"]:
        blockers.append("current_checkpoint_hash_missing_files")
    if required_missing:
        blockers.append("missing_required_fresh_or_strict_baseline_rows:" + ",".join(f"{r['model']}:{r['policy_id']}:{r['seq']}" for r in required_missing))
    if imported_not_fresh:
        blockers.append("imported_v118_reference_rows_require_v119_fresh_rerun_or_strict_current_code_equivalence")
    if stale_process_rows:
        blockers.append("candidate_stale_processes_require_manual_review")
    return {
        "schema": "acl2_v119tf_stage0_summary_v1",
        "generated_at_utc": generated_at,
        "stage0_gate_pass": not blockers,
        "stage0_blockers": blockers,
        "current_code_hashes": rel(STAGE0 / "current_code_hashes.json"),
        "current_checkpoint_hashes": rel(STAGE0 / "current_checkpoint_hashes.json"),
        "current_config_hashes": rel(STAGE0 / "current_config_hashes.json"),
        "canonical_baselines": rel(STAGE0 / "canonical_baselines.csv"),
        "frozen_negative_boundaries": rel(STAGE0 / "frozen_negative_boundaries.csv"),
        "frozen_strong_carriers": rel(STAGE0 / "frozen_strong_carriers.csv"),
        "stale_artifact_audit": rel(STAGE0 / "stale_artifact_audit.csv"),
        "canonical_schema": rel(STAGE0 / "V119_CANONICAL_RESULT_SCHEMA.md"),
        "baseline_row_count": len(baselines),
        "missing_required_baseline_count": len(required_missing),
        "imported_not_fresh_baseline_count": len(imported_not_fresh),
        "truthfulness_boundary": "Stage0 gate passes only when blockers is empty; no Stage1 or global v119 success is implied.",
    }


def main() -> int:
    generated_at = datetime.now(timezone.utc).isoformat()
    STAGE0.mkdir(parents=True, exist_ok=True)

    print("[stage0] hashing code files", flush=True)
    code_doc = current_code_hashes(generated_at)
    write_json(STAGE0 / "current_code_hashes.json", code_doc)
    print("[stage0] hashing config files", flush=True)
    config_doc = current_config_hashes(generated_at)
    write_json(STAGE0 / "current_config_hashes.json", config_doc)
    checkpoint_doc = current_checkpoint_hashes(generated_at)
    write_json(STAGE0 / "current_checkpoint_hashes.json", checkpoint_doc)

    hash_lookup = source_hash_lookup(code_doc, config_doc, checkpoint_doc)
    code_hash = str(code_doc["combined_code_hash"])
    config_hash = str(config_doc["combined_config_hash"])
    checkpoint_hash = str(checkpoint_doc["combined_checkpoint_hash"])

    print("[stage0] building canonical rows", flush=True)
    baselines = build_baseline_rows(generated_at, hash_lookup, code_hash, config_hash)
    negative = frozen_negative_boundaries()
    strong = frozen_strong_carriers()
    stale = stale_artifact_audit(generated_at)
    summary = stage0_summary(generated_at, code_doc, config_doc, checkpoint_doc, baselines, stale)

    write_csv(STAGE0 / "canonical_baselines.csv", baselines)
    write_csv(STAGE0 / "frozen_negative_boundaries.csv", negative)
    write_csv(STAGE0 / "frozen_strong_carriers.csv", strong)
    write_csv(STAGE0 / "stale_artifact_audit.csv", stale)
    write_text(STAGE0 / "V119_CANONICAL_RESULT_SCHEMA.md", canonical_schema_doc())
    write_json(STAGE0 / "stage0_summary.json", summary)

    write_csv(RESULT_ROOT / "V119_RUN_REGISTRY.csv", run_registry_rows(generated_at, code_hash, config_hash, checkpoint_hash))
    write_csv(RESULT_ROOT / "V119_CANONICAL_METRIC_ROWS.csv", build_global_metric_rows(baselines, checkpoint_hash))
    write_csv(RESULT_ROOT / "V119_BRANCH_COMPLETION_MATRIX.csv", branch_completion_rows(generated_at))
    write_code_audit(generated_at)
    write_initial_final_docs(generated_at, bool(summary["stage0_gate_pass"]), list(summary["stage0_blockers"]))

    print(json.dumps(clean_json(summary), indent=2, sort_keys=True, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
