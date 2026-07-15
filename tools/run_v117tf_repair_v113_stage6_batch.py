#!/usr/bin/env python3
"""Rebuild v113-HS Stage6 artifacts needed by the v117 Stage0 raw gate."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
V113_ROOT = ROOT / "results/acl2_v113hs_horizonstream_semantic_aware_geometric_evidence_influence"
HS_ROOT = ROOT / "third_party/HorizonStream"
IMAGE_ROOT = V113_ROOT / "kitti_generalizable"
SEMANTIC_ROOT = V113_ROOT / "semantic_projection"
CHECKPOINT = "checkpoints/HorizonStream.pt"
CONFIG = "configs/horizonstream_infer.yaml"
SEQS = ("00", "02")


@dataclass(frozen=True)
class Case:
    case_id: str
    role: str
    candidate_name: str
    output_prefix: str
    action: str
    control: str = ""


CASES = (
    Case(
        case_id="target_norm05_plus_m1",
        role="target",
        candidate_name="HS_L3_tiny_norm05_plus_HS_M1_default",
        output_prefix="stage6_hs_l3_tiny_norm05_plus_m1_full",
        action="HS_L3_value_stable_boost_tiny_norm05_plus_HS_M1_mrt_feature_risk_gate_mild",
    ),
    Case(
        case_id="ctrl_semantic_shuffle",
        role="control",
        candidate_name="ctrl_semantic_shuffle",
        output_prefix="stage6_hs_l3_tiny_norm05_plus_m1_ctrl_semantic_shuffle_full",
        action="HS_L3_value_stable_boost_tiny_norm05_plus_HS_M1_mrt_feature_risk_gate_mild",
        control="semantic_shuffle_by_frame",
    ),
    Case(
        case_id="ctrl_same_count_high_risk",
        role="control",
        candidate_name="ctrl_same_count_high_risk_frame_random",
        output_prefix="stage6_hs_l3_tiny_norm05_plus_m1_ctrl_same_count_high_risk_frame_random_full",
        action="HS_L3_value_stable_boost_tiny_norm05_plus_HS_M1_mrt_feature_risk_gate_mild",
        control="same_count_high_risk_frame_random",
    ),
    Case(
        case_id="ctrl_role_rotation",
        role="control",
        candidate_name="ctrl_role_rotation_dynamic_stable",
        output_prefix="stage6_hs_l3_tiny_norm05_plus_m1_ctrl_role_rotation_full",
        action="HS_L3_value_stable_boost_tiny_norm05_plus_HS_M1_mrt_feature_risk_gate_mild",
        control="role_rotation_dynamic_stable",
    ),
    Case(
        case_id="ctrl_low_risk_reverse",
        role="control",
        candidate_name="ctrl_low_risk_reverse",
        output_prefix="stage6_hs_l3_tiny_norm05_plus_m1_ctrl_low_risk_reverse_full",
        action="HS_L3_value_stable_boost_tiny_norm05_plus_HS_M1_mrt_feature_risk_gate_mild",
        control="low_risk_reverse",
    ),
    Case(
        case_id="repair_normpreserve_plus_m1",
        role="repair",
        candidate_name="HS_L3_tiny_normpreserve_plus_HS_M1_default",
        output_prefix="stage6_hs_l3_tiny_normpreserve_plus_m1_full",
        action="HS_L3_value_stable_boost_tiny_normpreserve_plus_HS_M1_mrt_feature_risk_gate_mild",
    ),
    Case(
        case_id="repair_l5_centered_plus_m1",
        role="repair",
        candidate_name="HS_L5_stable_centered_tiny_plus_HS_M1_default",
        output_prefix="stage6_hs_l5_stable_centered_tiny_plus_m1_full",
        action="HS_L5_value_stable_centered_tiny_plus_HS_M1_mrt_feature_risk_gate_mild",
    ),
    Case(
        case_id="ablation_semantic_shuffle_no_m1",
        role="ablation",
        candidate_name="HS_L3_tiny_norm05_ctrl_semantic_shuffle_no_M1",
        output_prefix="stage6_hs_l3_tiny_norm05_ctrl_semantic_shuffle_full",
        action="HS_L3_value_stable_boost_tiny_norm05",
        control="semantic_shuffle_by_frame",
    ),
)


SUMMARY_PREFIXES = {
    "target_norm05_plus_m1": "stage6_hs_l3_tiny_norm05_plus_m1_full_default",
    "ctrl_semantic_shuffle": "stage6_hs_l3_tiny_norm05_plus_m1_ctrl_semantic_shuffle_full_default",
    "ctrl_same_count_high_risk": "stage6_hs_l3_tiny_norm05_plus_m1_ctrl_same_count_high_risk_frame_random_full_default",
    "ctrl_role_rotation": "stage6_hs_l3_tiny_norm05_plus_m1_ctrl_role_rotation_full_default",
    "ctrl_low_risk_reverse": "stage6_hs_l3_tiny_norm05_plus_m1_ctrl_low_risk_reverse_full_default",
    "repair_normpreserve_plus_m1": "stage6_hs_l3_tiny_normpreserve_plus_m1_full_default",
    "repair_l5_centered_plus_m1": "stage6_hs_l5_stable_centered_tiny_plus_m1_full_default",
    "ablation_semantic_shuffle_no_m1": "stage6_hs_l3_tiny_norm05_ctrl_semantic_shuffle_full_default",
}


RESULT_FIELDS = [
    "schema",
    "job_id",
    "case_id",
    "role",
    "seq",
    "gpu",
    "action",
    "control",
    "output_root",
    "log_path",
    "command",
    "skipped_existing_success",
    "returncode",
    "duration_sec",
    "ran_eval",
    "ate_rmse",
    "num_pose_pairs",
    "sim3_scale",
    "stdout_tail",
]


def parse_csv_list(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def append_csv(path: Path, row: dict[str, Any], lock: threading.Lock) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with lock:
        exists = path.exists()
        with path.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS, extrasaction="ignore")
            if not exists:
                writer.writeheader()
            writer.writerow(row)


def append_jsonl(path: Path, row: dict[str, Any], lock: threading.Lock) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with lock:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def output_root_for(case: Case | None, seq: str, max_frames: int) -> Path:
    if case is None:
        name = f"baseline_kitti_{seq}"
    else:
        name = f"{case.output_prefix}_kitti_{seq}"
    if max_frames > 0:
        name = f"v117_repair_{name}_max{max_frames}"
    return V113_ROOT / "outputs" / name


def pipeline_summary(output_root: Path, seq: str) -> dict[str, Any]:
    path = output_root / "pipeline_summary.json"
    if not path.exists():
        return {"ran_eval": False}
    payload = json.loads(path.read_text(encoding="utf-8"))
    metrics = (
        payload.get("infer_eval", {})
        .get("sequences", {})
        .get(f"{seq}/02", {})
        .get("metrics", {})
        .get("main", {})
    )
    return {
        "ran_eval": bool(payload.get("ran_eval")),
        "ate_rmse": metrics.get("ate_rmse"),
        "num_pose_pairs": metrics.get("num_pose_pairs"),
        "sim3_scale": metrics.get("sim3_scale"),
    }


def make_command(job: dict[str, Any]) -> list[str]:
    cmd = [
        sys.executable,
        "run_pipeline.py",
        "--config",
        CONFIG,
        "--img-path",
        str(IMAGE_ROOT.resolve()),
        "--seq-list",
        job["seq"],
        "--camera",
        "02",
        "--checkpoint",
        CHECKPOINT,
        "--output-root",
        str(Path(job["output_root"]).resolve()),
        "--no-camera-preprocess",
        "--offload-outputs-to-cpu",
        "--no-save-videos",
        "--no-save-points",
        "--no-save-images",
        "--no-save-depth",
        "--no-save-depth-conf",
        "--no-mask-sky",
        "--no-point-mask-sky",
        "--no-loop",
        "--eval-pose-variants",
        "main",
    ]
    if int(job["max_frames"]) > 0:
        cmd.extend(["--max-frames", str(int(job["max_frames"]))])
    return cmd


def command_for_log(job: dict[str, Any], cmd: list[str]) -> str:
    env_parts = [
        "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True",
        f"CUDA_VISIBLE_DEVICES={job['gpu']}",
        f"HS_V113_SEMANTIC_ROOT={SEMANTIC_ROOT.resolve()}",
    ]
    if job["action"]:
        env_parts.append(f"HS_V113_ACTION={job['action']}")
    if job["control"]:
        env_parts.append(f"HS_V113_CONTROL={job['control']}")
    return " ".join(env_parts + cmd)


def run_job(job: dict[str, Any]) -> dict[str, Any]:
    started = time.time()
    output_root = Path(job["output_root"])
    log_path = Path(job["log_path"])
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = make_command(job)
    command = command_for_log(job, cmd)

    if job["skip_existing_success"] and pipeline_summary(output_root, job["seq"]).get("ran_eval"):
        summary = pipeline_summary(output_root, job["seq"])
        return {
            "schema": "acl2_v117tf_repair_v113_stage6_run_result_v1",
            **{key: job.get(key, "") for key in ["job_id", "case_id", "role", "seq", "gpu", "action", "control"]},
            "output_root": str(output_root),
            "log_path": str(log_path),
            "command": command,
            "skipped_existing_success": True,
            "returncode": 0,
            "duration_sec": round(time.time() - started, 3),
            "ran_eval": summary.get("ran_eval"),
            "ate_rmse": summary.get("ate_rmse"),
            "num_pose_pairs": summary.get("num_pose_pairs"),
            "sim3_scale": summary.get("sim3_scale"),
            "stdout_tail": "",
        }

    env = os.environ.copy()
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    env["CUDA_VISIBLE_DEVICES"] = str(job["gpu"])
    env["HS_V113_SEMANTIC_ROOT"] = str(SEMANTIC_ROOT.resolve())
    env.pop("HS_V113_ACTION", None)
    env.pop("HS_V113_CONTROL", None)
    if job["action"]:
        env["HS_V113_ACTION"] = str(job["action"])
    if job["control"]:
        env["HS_V113_CONTROL"] = str(job["control"])

    tail: deque[str] = deque(maxlen=120)
    with log_path.open("w", encoding="utf-8") as log:
        log.write("# v117 repair v113 Stage6 command\n")
        log.write(f"cwd={HS_ROOT}\n")
        log.write(command + "\n\n")
        log.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=str(HS_ROOT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            log.write(line)
            tail.append(line)
        returncode = proc.wait()

    summary = pipeline_summary(output_root, job["seq"])
    return {
        "schema": "acl2_v117tf_repair_v113_stage6_run_result_v1",
        **{key: job.get(key, "") for key in ["job_id", "case_id", "role", "seq", "gpu", "action", "control"]},
        "output_root": str(output_root),
        "log_path": str(log_path),
        "command": command,
        "skipped_existing_success": False,
        "returncode": int(returncode),
        "duration_sec": round(time.time() - started, 3),
        "ran_eval": summary.get("ran_eval"),
        "ate_rmse": summary.get("ate_rmse"),
        "num_pose_pairs": summary.get("num_pose_pairs"),
        "sim3_scale": summary.get("sim3_scale"),
        "stdout_tail": "".join(tail)[-4000:],
    }


def run_gpu_queue(
    gpu: str,
    rows: list[dict[str, Any]],
    results_csv: Path,
    results_jsonl: Path,
    csv_lock: threading.Lock,
    jsonl_lock: threading.Lock,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        result = run_job(row)
        append_csv(results_csv, result, csv_lock)
        append_jsonl(results_jsonl, result, jsonl_lock)
        print(
            "RETURN "
            f"rc={result['returncode']} gpu={gpu} job={result['job_id']} "
            f"seq={result['seq']} skipped={result['skipped_existing_success']} "
            f"duration_sec={result['duration_sec']} ate_rmse={result.get('ate_rmse')}",
            flush=True,
        )
        if int(result["returncode"]) != 0:
            print(result["stdout_tail"][-2000:], flush=True)
        out.append(result)
    return out


def build_jobs(gpus: list[str], max_frames: int, skip_existing_success: bool) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    log_suffix = f"_max{max_frames}" if max_frames > 0 else ""
    for seq in SEQS:
        jobs.append(
            {
                "schema": "acl2_v117tf_repair_v113_stage6_job_v1",
                "job_id": f"baseline_{seq}",
                "case_id": "baseline",
                "role": "baseline",
                "seq": seq,
                "action": "",
                "control": "",
                "output_root": str(output_root_for(None, seq, max_frames)),
                "log_path": str(V113_ROOT / "logs" / f"v117_repair_stage6_baseline_kitti_{seq}{log_suffix}.log"),
                "max_frames": int(max_frames),
                "skip_existing_success": skip_existing_success,
            }
        )
        for case in CASES:
            jobs.append(
                {
                    "schema": "acl2_v117tf_repair_v113_stage6_job_v1",
                    "job_id": f"{case.case_id}_{seq}",
                    "case_id": case.case_id,
                    "role": case.role,
                    "seq": seq,
                    "action": case.action,
                    "control": case.control,
                    "output_root": str(output_root_for(case, seq, max_frames)),
                    "log_path": str(
                        V113_ROOT / "logs" / f"v117_repair_stage6_{case.output_prefix}_kitti_{seq}{log_suffix}.log"
                    ),
                    "max_frames": int(max_frames),
                    "skip_existing_success": skip_existing_success,
                }
            )
    for idx, job in enumerate(jobs):
        job["gpu"] = gpus[idx % len(gpus)]
    return jobs


def run_batch(args: argparse.Namespace, jobs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    manifest_path = V113_ROOT / "diagnostics/v117_repair_stage6_manifest.csv"
    write_csv(
        manifest_path,
        jobs,
        [
            "schema",
            "job_id",
            "case_id",
            "role",
            "seq",
            "gpu",
            "action",
            "control",
            "output_root",
            "log_path",
            "max_frames",
            "skip_existing_success",
        ],
    )
    results_csv = V113_ROOT / "diagnostics/v117_repair_stage6_run_results.csv"
    results_jsonl = V113_ROOT / "diagnostics/v117_repair_stage6_run_results.jsonl"
    if args.fresh_results:
        results_csv.unlink(missing_ok=True)
        results_jsonl.unlink(missing_ok=True)

    queues: dict[str, list[dict[str, Any]]] = {}
    for job in jobs:
        queues.setdefault(str(job["gpu"]), []).append(job)

    print(
        f"planned_jobs={len(jobs)} queues={len(queues)} manifest={manifest_path} "
        f"results_csv={results_csv}",
        flush=True,
    )
    csv_lock = threading.Lock()
    jsonl_lock = threading.Lock()
    all_results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, min(args.max_workers, len(queues)))) as executor:
        futures = {
            executor.submit(run_gpu_queue, gpu, rows, results_csv, results_jsonl, csv_lock, jsonl_lock): gpu
            for gpu, rows in queues.items()
        }
        for future in as_completed(futures):
            all_results.extend(future.result())
    failures = [row for row in all_results if int(row["returncode"]) != 0]
    print(f"batch_done results={len(all_results)} failures={len(failures)}", flush=True)
    return all_results


def run_logged_command(command: list[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write("# command\n")
        log.write(" ".join(command) + "\n\n")
        log.flush()
        proc = subprocess.Popen(
            command,
            cwd=str(ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            log.write(line)
        return proc.wait()


def summarize() -> None:
    for case in CASES:
        output_prefix = SUMMARY_PREFIXES[case.case_id]
        command = [
            sys.executable,
            "tools/build_v113hs_action_metric_summary.py",
            "--results-root",
            str(V113_ROOT),
            "--seqs",
            ",".join(SEQS),
            "--baseline-name",
            "default_baseline",
            "--candidate-name",
            case.candidate_name,
            "--baseline-template",
            "outputs/baseline_kitti_{seq}",
            "--candidate-template",
            f"outputs/{case.output_prefix}_kitti_{{seq}}",
            "--output-prefix",
            output_prefix,
            "--claim-boundary",
            "v117 repair reconstruction of the v113 Stage6 default full-sequence artifact; not a new v117 runtime promotion.",
        ]
        rc = run_logged_command(command, V113_ROOT / "logs" / f"v117_repair_summarize_{output_prefix}.log")
        if rc != 0:
            raise SystemExit(rc)

    decision_command = [
        sys.executable,
        "tools/build_v113hs_stage6_decision_summary.py",
        "--results-root",
        str(V113_ROOT),
    ]
    rc = run_logged_command(decision_command, V113_ROOT / "logs/v117_repair_stage6_decision_summary.log")
    if rc != 0:
        raise SystemExit(rc)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", default="0,1,2,3,4")
    parser.add_argument("--max-workers", type=int, default=5)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--skip-existing-success", action="store_true")
    parser.add_argument("--fresh-results", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--run-only", action="store_true")
    args = parser.parse_args()

    if args.max_frames > 0 and not args.run_only:
        raise SystemExit("--max-frames is only supported with --run-only, because summary inputs expect full output names.")

    gpus = parse_csv_list(args.gpus)
    if not gpus:
        raise SystemExit("No GPUs supplied.")
    if not args.summarize_only:
        jobs = build_jobs(gpus, int(args.max_frames), bool(args.skip_existing_success))
        results = run_batch(args, jobs)
        failures = [row for row in results if int(row["returncode"]) != 0]
        if failures:
            raise SystemExit(1)
    if not args.run_only:
        summarize()


if __name__ == "__main__":
    main()
