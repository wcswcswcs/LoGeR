#!/usr/bin/env python3
"""Freeze v108 evidence for ACL2 v109TF F-surface causal dissection."""

from __future__ import annotations

import csv
import json
import math
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "results/acl2_v109tf_lingbot_f_surface_causal_dissection_semantic_memory_control"
OUT = RESULT_ROOT / "stage0_evidence_freeze"

V105 = ROOT / "results/acl2_v105tf_dualtrack_lingbot_loger_evidence_eligibility"
V108 = ROOT / "results/acl2_v108tf_lingbot_memory_operation_action_surface_search"
V108_STAGE2 = V108 / "stage2_semantic_cue_bank"
V108_STAGE3 = V108 / "stage3_operation_cue_screen"
V108_STAGE4 = V108 / "stage4_full_kitti_00_02_action_pilot"
V108_STAGE5 = V108 / "stage5_full_kitti_00_01_02_05_validation"

EXPECTED_BASELINE = {
    "00": 46.00057328153847,
    "01": 57.097417656974685,
    "02": 77.48077587398916,
    "05": 19.961256567907505,
}


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
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
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
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return rel(value)
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
        "ACL2_V108_STAGE4_POLICY_ID",
        "ACL2_V109",
        "run_v107r_stage6_semantic_wrapper_policy_manifest.py",
        "run_v108tf_gpu_serial_policy_manifest.py",
    )
    self_markers = (
        "build_v109tf_stage0_evidence_freeze.py",
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
        ("v105_full_kitti_baseline_table", V105 / "stage1_lingbot_baseline/full_sequence_metrics/lingbot_stream_default_full_metrics.csv", "required"),
        ("v108_stage2_summary", V108_STAGE2 / "stage2_summary.json", "required"),
        ("v108_stage2_frame_semantic_summary", V108_STAGE2 / "frame_semantic_summary.csv", "required"),
        ("v108_stage2_operation_semantic_summary", V108_STAGE2 / "operation_semantic_summary.csv", "required"),
        ("v108_stage3_surface_policy_rows", V108_STAGE3 / "surface_policy_rows.csv", "required"),
        ("v108_stage3_surface_policy_frame_rows", V108_STAGE3 / "surface_policy_frame_rows.csv", "required"),
        ("v108_stage4_summary", V108_STAGE4 / "stage4_summary.json", "required"),
        ("v108_stage4_full_metric_rows", V108_STAGE4 / "full_sequence_metric_rows.csv", "required"),
        ("v108_stage4_action_fidelity_rows", V108_STAGE4 / "action_fidelity_rows.csv", "required"),
        ("v108_stage4_semantic_control_rows", V108_STAGE4 / "semantic_control_rows.csv", "required"),
        ("v108_stage5_summary", V108_STAGE5 / "stage5_summary.json", "required"),
        ("v108_stage5_action_config_rows", V108_STAGE5 / "action_config_rows.csv", "required"),
        ("v108_stage5_keyframe_snap_rows", V108_STAGE5 / "keyframe_snap_rows.csv", "required"),
        ("v108_stage5_full_metric_rows", V108_STAGE5 / "full_sequence_metric_rows.csv", "required"),
        ("v108_stage5_rolling_metric_rows", V108_STAGE5 / "rolling_metric_rows.csv", "required"),
        ("v108_stage5_local_handoff_metric_rows", V108_STAGE5 / "local_handoff_metric_rows.csv", "required"),
        ("v108_stage5_action_fidelity_rows", V108_STAGE5 / "action_fidelity_rows.csv", "required"),
        ("v108_stage5_semantic_control_rows", V108_STAGE5 / "semantic_control_rows.csv", "required"),
        ("v108_stage5_run_manifest", V108_STAGE5 / "run_manifest.csv", "required"),
        ("v108_stage5_run_results", V108_STAGE5 / "run_results.csv", "required"),
        ("v108_stage5_no_action_control_rows", V108_STAGE5 / "no_action_control_rows.csv", "required"),
    ]
    rows: list[dict[str, Any]] = []
    for artifact_id, path, requirement in artifacts:
        row_count: int | str = ""
        if path.exists() and path.suffix == ".csv":
            row_count = len(read_csv(path))
        rows.append(
            {
                "schema": "acl2_v109tf_stage0_artifact_manifest_row_v1",
                "artifact_id": artifact_id,
                "path": rel(path),
                "requirement": requirement,
                "exists": path.exists(),
                "suffix": path.suffix,
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
        ate = fnum(row.get("ATE_full_sim3_m"))
        exact[seq] = {
            "observed": ate,
            "expected": EXPECTED_BASELINE[seq],
            "abs_diff": abs(ate - EXPECTED_BASELINE[seq]) if math.isfinite(ate) else float("nan"),
            "match": math.isfinite(ate) and abs(ate - EXPECTED_BASELINE[seq]) <= 1e-12,
        }
        rows.append(
            {
                "schema": "acl2_v109tf_stage0_full_kitti_baseline_row_v1",
                "seq": seq,
                "dataset": row.get("dataset", ""),
                "method": row.get("method", ""),
                "frames": row.get("frames", ""),
                "ATE_full_sim3_m": row.get("ATE_full_sim3_m", ""),
                "final_error_m": row.get("final_error_m", ""),
                "benchmark_rpe_trans": row.get("benchmark_rpe_trans", ""),
                "benchmark_rpe_rot": row.get("benchmark_rpe_rot", ""),
                "rolling_ATE_mean": row.get("rolling_ATE_mean", ""),
                "rolling_ATE_p90": row.get("rolling_ATE_p90", ""),
                "rolling_ATE_max": row.get("rolling_ATE_max", ""),
                "rolling_worse_fraction_gt_0p05": row.get("rolling_worse_fraction_gt_0p05", ""),
                "local_window_ATE_median": row.get("local_window_ATE_median", ""),
                "source": rel(src),
            }
        )
    return rows, exact


def surface_rows(surface: str) -> list[dict[str, Any]]:
    full_rows = read_csv(V108_STAGE5 / "full_sequence_metric_rows.csv")
    action_rows = {
        (row.get("policy_id", ""), row.get("seq", "")): row
        for row in read_csv(V108_STAGE5 / "action_fidelity_rows.csv")
    }
    out: list[dict[str, Any]] = []
    for row in full_rows:
        if row.get("surface_id") != surface:
            continue
        fid = action_rows.get((row.get("policy_id", ""), row.get("seq", "")), {})
        out.append(
            {
                "schema": f"acl2_v109tf_stage0_{surface.lower()}_surface_metric_row_v1",
                "surface_id": surface,
                "policy_id": row.get("policy_id", ""),
                "policy_family": row.get("policy_family", ""),
                "seq": row.get("seq", ""),
                "full_ATE_sim3": row.get("full_ATE_sim3", ""),
                "baseline_full_ATE_sim3": row.get("baseline_full_ATE_sim3", ""),
                "full_ATE_sim3_delta_action_minus_baseline": row.get("full_ATE_sim3_delta_action_minus_baseline", ""),
                "full_ATE_sim3_relative_improvement_vs_baseline": row.get("full_ATE_sim3_relative_improvement_vs_baseline", ""),
                "final_error_relative_improvement_vs_baseline": row.get("final_error_relative_improvement_vs_baseline", ""),
                "local_window_ATE_rel_improvement_vs_baseline_median": row.get("local_window_ATE_rel_improvement_vs_baseline_median", ""),
                "action_fidelity_pass": row.get("action_fidelity_pass", ""),
                "expected_action_frame_count": fid.get("expected_action_frame_count", ""),
                "action_effective_frame_count": fid.get("action_effective_frame_count", ""),
                "action_noop_frame_count": fid.get("action_noop_frame_count", ""),
                "action_file": fid.get("action_file", ""),
                "source": rel(V108_STAGE5 / "full_sequence_metric_rows.csv"),
            }
        )
    return out


def known_facts(baseline_rows: list[dict[str, Any]], baseline_exact: dict[str, Any]) -> dict[str, Any]:
    stage2 = read_json(V108_STAGE2 / "stage2_summary.json")
    stage3 = read_json(V108_STAGE3 / "stage3_summary.json")
    stage4 = read_json(V108_STAGE4 / "stage4_summary.json")
    stage5 = read_json(V108_STAGE5 / "stage5_summary.json")
    f_rows = [row for row in surface_rows("F") if row["policy_family"] == "semantic_plus_internal"]
    e_rows = [row for row in surface_rows("E") if row["policy_family"] == "semantic_plus_internal"]
    return {
        "schema": "acl2_v109tf_stage0_v108_known_facts_v1",
        "baseline_ate_by_seq": {row["seq"]: row["ATE_full_sim3_m"] for row in baseline_rows},
        "baseline_exact_match_by_seq": baseline_exact,
        "v108_stage2": {
            "stage2_pass": stage2.get("stage2_pass"),
            "expected_frame_count": stage2.get("expected_frame_count"),
            "processed_frame_count": stage2.get("processed_frame_count"),
            "frame_semantic_coverage": stage2.get("frame_semantic_coverage"),
            "token_semantic_row_count": stage2.get("token_semantic_row_count"),
            "operation_row_count": stage2.get("operation_row_count"),
            "operation_rows_join_coverage_mean": stage2.get("operation_rows_join_coverage_mean"),
            "semantic_nonvoid_frame_ratio": stage2.get("semantic_nonvoid_frame_ratio"),
            "semantic_nonvoid_frame_ratio_ge_0p95": stage2.get("semantic_nonvoid_frame_ratio_ge_0p95"),
            "semantic_patch_nonvoid_ratio": stage2.get("semantic_patch_nonvoid_ratio"),
            "semantic_patch_purity_mean": stage2.get("semantic_patch_purity_mean"),
            "surface_row_counts": stage2.get("surface_row_counts"),
        },
        "v108_stage3": {
            "stage3_pass": stage3.get("stage3_pass"),
            "full_sequence_candidate_policy_ids": stage3.get("full_sequence_candidate_policy_ids", []),
            "surfaces_stage1_new_hook_needed_or_not_allowed": stage3.get("surfaces_stage1_new_hook_needed_or_not_allowed", []),
            "top_candidate_by_surface": stage3.get("top_candidate_by_surface", {}),
        },
        "v108_stage4": {
            "stage4_pass": stage4.get("stage4_pass"),
            "metric_complete": stage4.get("metric_complete"),
            "passing_surfaces": stage4.get("passing_surfaces", []),
            "semantic_control_rows": stage4.get("semantic_control_rows", []),
        },
        "v108_stage5": {
            "stage5_pass": stage5.get("stage5_pass"),
            "metric_complete": stage5.get("metric_complete"),
            "blocker": stage5.get("blocker"),
            "passing_surfaces": stage5.get("passing_surfaces", []),
            "observed_run_worker_count": stage5.get("observed_run_worker_count"),
            "observed_run_worker_historical_count": stage5.get("observed_run_worker_historical_count"),
            "observed_run_worker_historical_failure_count": stage5.get("observed_run_worker_historical_failure_count"),
            "semantic_control_rows": stage5.get("semantic_control_rows", []),
            "f_semantic_plus_rows": f_rows,
            "e_semantic_plus_rows": e_rows,
        },
        "runtime_cue_boundary": [
            "semantic label/confidence/purity/boundary/role/seed continuity",
            "LingBot internal operation type/context/token/keyframe/cache lifecycle/source age/special-token path/action fidelity",
        ],
        "forbidden_runtime_cues": ["external depth model", "MoGe", "GT", "SLAM", "post-hoc Sim3"],
    }


def forbidden_repeat_text() -> str:
    return """# ACL2 v109TF Forbidden Repeat List

1. 只跑 selected window / 96F / trace32 就 claim geometry improvement。
2. 只看 L3 不看 full KITTI ATE。
3. 只用 semantic+internal，不跑 internal_only / shuffle / random controls。
4. 继续 headlocal relaxed context-only demotion 作为默认路线。
5. 用 E surface 的 00 improvement 掩盖 01/05 harm。
6. 用 full ATE improvement 但 semantic controls 不过来 claim semantic-aware method。
7. 使用外部 depth model、MoGe、GT 或 SLAM 作为 runtime cue。
"""


def allowed_next_steps_text() -> str:
    return """# ACL2 v109TF Action Surface Allowed Next Steps

- F is the primary route: causal dissection, full KITTI controls, then hard-negative safety repair if needed.
- E is a high-gain/high-risk control only unless guarded by an F-derived safety rule and revalidated.
- A/B are schedule-sensitivity baselines, not semantic-aware methods.
- C/D are fallback minimal hooks only after F semantic causality or safety fails; they require no-action parity before full pilots.
- Any runtime action claim must use full KITTI ATE plus internal/shuffle/random controls.
- GT, external depth, SLAM, and post-hoc Sim3 remain offline-evaluation only and cannot be runtime cues.
"""


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    manifest_rows = artifact_manifest()
    baseline_rows, baseline_exact = baseline_table()
    pending = process_rows()
    required_missing = [row["artifact_id"] for row in manifest_rows if row["requirement"] == "required" and not row["exists"]]
    baseline_exact_pass = set(baseline_exact) == set(EXPECTED_BASELINE) and all(row["match"] for row in baseline_exact.values())
    stage2 = read_json(V108_STAGE2 / "stage2_summary.json")
    stage5 = read_json(V108_STAGE5 / "stage5_summary.json")
    stage2_pass = bool(stage2.get("stage2_pass", False))
    stage5_complete = bool(stage5.get("metric_complete", False)) and int(stage5.get("observed_run_worker_count", 0)) == 32
    stage0_pass = baseline_exact_pass and stage2_pass and stage5_complete and not required_missing and not pending

    f_rows = surface_rows("F")
    e_rows = surface_rows("E")
    facts = known_facts(baseline_rows, baseline_exact)

    write_csv(OUT / "available_artifact_manifest.csv", manifest_rows)
    write_csv(OUT / "full_kitti_baseline_table.csv", baseline_rows)
    write_csv(OUT / "f_surface_baseline_rows.csv", f_rows)
    write_csv(OUT / "e_surface_control_rows.csv", e_rows)
    write_json(OUT / "v108_known_facts.json", facts)
    write_text(OUT / "forbidden_repeat_list.md", forbidden_repeat_text())
    write_text(OUT / "action_surface_allowed_next_steps.md", allowed_next_steps_text())

    outputs = {
        "stage0_summary": rel(OUT / "stage0_summary.json"),
        "v108_known_facts": rel(OUT / "v108_known_facts.json"),
        "f_surface_baseline_rows": rel(OUT / "f_surface_baseline_rows.csv"),
        "e_surface_control_rows": rel(OUT / "e_surface_control_rows.csv"),
        "full_kitti_baseline_table": rel(OUT / "full_kitti_baseline_table.csv"),
        "forbidden_repeat_list": rel(OUT / "forbidden_repeat_list.md"),
        "action_surface_allowed_next_steps": rel(OUT / "action_surface_allowed_next_steps.md"),
        "available_artifact_manifest": rel(OUT / "available_artifact_manifest.csv"),
    }
    summary = {
        "schema": "acl2_v109tf_stage0_summary_v1",
        "stage0_pass": stage0_pass,
        "baseline_exact_pass": baseline_exact_pass,
        "baseline_exact_match_by_seq": baseline_exact,
        "v108_stage2_pass": stage2_pass,
        "v108_stage5_metric_complete": stage5_complete,
        "required_missing_artifacts": required_missing,
        "pending_lingbot_process_rows": pending,
        "f_surface_row_count": len(f_rows),
        "e_surface_row_count": len(e_rows),
        "forbidden_repeat_list_written": (OUT / "forbidden_repeat_list.md").exists(),
        "outputs": outputs,
    }
    write_json(OUT / "stage0_summary.json", summary)
    if not stage0_pass:
        write_text(
            OUT / "STAGE0_EVIDENCE_FREEZE_BLOCKED.md",
            "# ACL2 v109TF Stage0 Evidence Freeze Blocked\n\n"
            f"- required_missing_artifacts: `{required_missing}`\n"
            f"- pending_lingbot_process_rows: `{pending}`\n"
            f"- baseline_exact_pass: `{baseline_exact_pass}`\n"
            f"- v108_stage2_pass: `{stage2_pass}`\n"
            f"- v108_stage5_metric_complete: `{stage5_complete}`\n",
        )
    print(json.dumps(clean_json(summary), indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
