#!/usr/bin/env python3
"""Unified wrapper for the v105 persistent Phase6 batch speed branch.

This runner keeps the already verified Phase6 chunk generation code in
probe_v105_phase6_reuse_models_batch.py and the scene assembly code in
run_v105_fullscene_multichunk_repair.py.  Its job is orchestration: launch one
persistent Phase6 process per GPU, run the final short chunk, assemble the
complete scene MP4/zip, and write a single audit summary for the whole run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path("/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python")
DEFAULT_CONFIG = "configs/v105/baseline_chunk_table/baseline_x_gapadaptive_sam2.generated.yaml"
DEFAULT_X0_SUMMARY = (
    "Stream3D/outputs/audit/v105_repair_min_scene0050_x0_20260712/"
    "baseline_x_sam2_twostage_sam2/summary.json"
)
DEFAULT_DURATION_MANIFEST = (
    "Stream3D/outputs/audit/v105_fullscene_skip_chunk_visuals_full931_20260712/"
    "fullscene_multichunk_manifest.jsonl"
)
DEFAULT_FROZEN_X1_SUMMARY_TEMPLATE = (
    "Stream3D/outputs/audit/v105_scene0050_fullscene_frame0seed_promptrepair_full931_20260712/"
    "x1seed_{short}_start{frame_start:04d}_f{frame_count}/baseline_x_gapadaptive_sam2/summary.json"
)
DEFAULT_FROZEN_BIRTH_RECORDS_TEMPLATE = (
    "Stream3D/outputs/audit/v105_scene0050_fullscene_frame0seed_promptrepair_full931_20260712/"
    "x1seed_{short}_start{frame_start:04d}_f{frame_count}/birth_bank/birth_records.json"
)
DEFAULT_FROZEN_ALLTRACKER_TEMPLATE = (
    "Stream3D/outputs/audit/v105_scene0050_fullscene_frame0seed_promptrepair_full931_20260712/"
    "alltracker_{short}_start{frame_start:04d}_f{frame_count}"
)


PHASE6_ARGS = [
    "--variant",
    "p4_anchor_period_definite",
    "--use-video-feature-bank",
    "--video-feature-bank-storage-device",
    "cuda",
    "--video-gpu-hot-window",
    "0",
    "--birth-anchor-period",
    "2",
    "--birth-anchor-offset",
    "1",
    "--birth-anchor-force-candidate-area-thresh",
    "50000",
    "--birth-anchor-force-max-events",
    "1",
    "--max-births-per-frame",
    "1",
    "--choice-policy",
    "max_candidate_support_valid_mask_per_point",
    "--min-component-area",
    "120",
    "--max-points-per-frame",
    "128",
    "--base-points-per-component",
    "1",
    "--area-per-extra-point",
    "40000",
    "--max-points-per-component",
    "8",
    "--min-candidate-touch-area",
    "16",
    "--min-candidate-touch-ratio",
    "0.0005",
    "--enable-frame0-residual-repair",
    "--scheduler-mode",
    "independent_anchor",
    "--stream-state-repair-mode",
    "reconsolidate",
    "--allow-missing-x0-diagnostics",
    "--allow-missing-x1-diagnostics",
    "--reuse-video-state-template",
    "--skip-chunk-overlays",
    "--skip-chunk-sheets",
    "--skip-chunk-video",
    "--prompt-repair-mode",
    "on_raw_or_filtered_empty",
    "--prompt-repair-min-component-area",
    "120",
    "--prompt-repair-max-components",
    "8",
    "--prompt-repair-positive-points-per-component",
    "3",
    "--prompt-repair-negative-points-per-component",
    "2",
    "--prompt-repair-box-expand-px",
    "6",
    "--prompt-repair-min-component-completion-ratio",
    "0.20",
    "--prompt-repair-pred-iou-thresh",
    "0.5",
    "--prompt-repair-stability-score-thresh",
    "0.5",
    "--frame0-residual-candidate-mode",
    "eroded_core_uncovered",
    "--frame0-residual-core-erosion-px",
    "7",
    "--frame0-residual-max-points",
    "160",
    "--frame0-residual-min-component-area",
    "300",
    "--frame0-residual-max-points-per-component",
    "16",
    "--frame0-residual-min-candidate-touch-ratio",
    "0.005",
    "--frame0-residual-max-existing-overlap-ratio",
    "0.98",
    "--frame0-residual-max-births",
    "4",
]


def rel(path: Path | str) -> str:
    p = Path(path)
    try:
        return p.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except Exception:
        return p.as_posix()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def numeric_stem(path: Path) -> int:
    try:
        return int(path.stem)
    except Exception:
        return 10**15


def scene_short(scene_id: str) -> str:
    return scene_id.split("_")[0]


def selected_frame_ids(scene_id: str, stride: int) -> list[int]:
    color_dir = REPO_ROOT / "Stream3D/data/scannet/processed" / scene_id / "color"
    files = sorted(color_dir.glob("*.jpg"), key=numeric_stem)
    if not files:
        files = sorted(color_dir.glob("*.png"), key=numeric_stem)
    ids = [numeric_stem(path) for path in files if numeric_stem(path) < 10**12]
    return ids[:: max(int(stride), 1)]


def chunk_specs(frame_ids: list[int], chunk_size: int) -> list[dict[str, int]]:
    specs: list[dict[str, int]] = []
    idx = 0
    while idx < len(frame_ids):
        count = min(int(chunk_size), len(frame_ids) - idx)
        specs.append({"start_index": idx, "frame_start": int(frame_ids[idx]), "frame_count": int(count)})
        idx += int(chunk_size)
    return specs


def load_duration_map(path: Path) -> dict[tuple[int, int], float]:
    durations: dict[tuple[int, int], float] = {}
    if not path.exists():
        return durations
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("event") == "step_end" and row.get("step_name") == "phase6_p6_candidate":
            durations[(int(row["frame_start"]), int(row["frame_count"]))] = float(row["runtime_sec"])
    return durations


def assign_jobs(jobs: list[dict[str, int]], gpus: list[str], durations: dict[tuple[int, int], float]) -> dict[str, list[dict[str, int]]]:
    bins: list[tuple[float, str, list[dict[str, int]]]] = [(0.0, gpu, []) for gpu in gpus]
    for job in sorted(jobs, key=lambda item: durations.get((item["frame_start"], item["frame_count"]), float(item["frame_count"])), reverse=True):
        bins.sort(key=lambda item: item[0])
        total, gpu, items = bins[0]
        items.append(job)
        bins[0] = (total + durations.get((job["frame_start"], job["frame_count"]), float(job["frame_count"])), gpu, items)
    return {gpu: sorted(items, key=lambda item: item["frame_start"]) for _, gpu, items in bins}


def run_logged(cmd: list[str], *, log_path: Path, env: dict[str, str]) -> dict[str, Any]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + " ".join(cmd) + "\n\n")
        log.flush()
        proc = subprocess.run(cmd, cwd=str(REPO_ROOT), env=env, stdout=log, stderr=subprocess.STDOUT, check=False)
    return {
        "cmd": cmd,
        "log_path": rel(log_path),
        "returncode": int(proc.returncode),
        "wall_sec": float(time.time() - t0),
    }


def launch_batch(
    *,
    gpu: str,
    starts: list[int],
    frame_count: int,
    output_root: Path,
    summary_name: str,
    log_path: Path,
    args: argparse.Namespace,
) -> subprocess.Popen[Any]:
    short = scene_short(str(args.scene_id))
    x1_summary_template = str(args.x1_summary_template).replace("{short}", short)
    birth_records_template = str(args.frame0_birth_records_template).replace("{short}", short)
    alltracker_template = str(args.alltracker_dir_template).replace("{short}", short)
    env = os.environ.copy()
    env.update({"CUDA_VISIBLE_DEVICES": str(gpu), "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True", "PYTHONUNBUFFERED": "1"})
    cmd = [
        str(PYTHON),
        "tools/probe_v105_phase6_reuse_models_batch.py",
        "--scene-id",
        str(args.scene_id),
        "--frame-starts",
        ",".join(str(start) for start in starts),
        "--frame-count",
        str(int(frame_count)),
        "--config",
        str(args.config),
        "--x0-summary",
        str(args.x0_summary),
        "--x1-summary-template",
        x1_summary_template,
        "--frame0-birth-records-template",
        birth_records_template,
        "--alltracker-dir-template",
        alltracker_template,
        "--output-root",
        rel(output_root),
        "--summary-name",
        summary_name,
        "--phase6-args",
        *PHASE6_ARGS,
    ]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = log_path.open("w", encoding="utf-8")
    log.write("$ " + " ".join(cmd) + "\n\n")
    log.flush()
    proc = subprocess.Popen(cmd, cwd=str(REPO_ROOT), env=env, stdout=log, stderr=subprocess.STDOUT)
    proc._v105_log_handle = log  # type: ignore[attr-defined]
    proc._v105_cmd = cmd  # type: ignore[attr-defined]
    proc._v105_log_path = log_path  # type: ignore[attr-defined]
    proc._v105_start_time = time.time()  # type: ignore[attr-defined]
    return proc


def collect_proc(proc: subprocess.Popen[Any]) -> dict[str, Any]:
    returncode = proc.wait()
    wall_sec = time.time() - float(proc._v105_start_time)  # type: ignore[attr-defined]
    proc._v105_log_handle.close()  # type: ignore[attr-defined]
    return {
        "cmd": proc._v105_cmd,  # type: ignore[attr-defined]
        "log_path": rel(proc._v105_log_path),  # type: ignore[attr-defined]
        "returncode": int(returncode),
        "wall_sec": float(wall_sec),
    }


def run(args: argparse.Namespace) -> None:
    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = REPO_ROOT / output_root
    if output_root.exists():
        if not bool(args.force):
            raise FileExistsError({"output_root": rel(output_root), "repair": "use --force or a fresh root"})
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    log_dir = output_root / "run_logs"

    frame_ids = selected_frame_ids(str(args.scene_id), int(args.frame_stride))
    if int(args.max_frames) > 0:
        frame_ids = frame_ids[: int(args.max_frames)]
    specs = chunk_specs(frame_ids, int(args.chunk_size))
    full_specs = [spec for spec in specs if spec["frame_count"] == int(args.chunk_size)]
    tail_specs = [spec for spec in specs if spec["frame_count"] != int(args.chunk_size)]
    gpus = [part.strip() for part in str(args.gpus).split(",") if part.strip()]
    if not gpus:
        raise ValueError("--gpus must contain at least one GPU id")
    durations = load_duration_map(REPO_ROOT / str(args.duration_manifest))
    assignments = assign_jobs(full_specs, gpus, durations)

    started = time.time()
    prepare_result: dict[str, Any] | None = None
    if bool(args.prepare_x1_alltracker):
        prepare_cmd = [
            str(PYTHON),
            "tools/run_v105_fullscene_multichunk_repair.py",
            "--scenes",
            str(args.scene_id),
            "--output-root",
            rel(output_root),
            "--gpus",
            str(args.gpus),
            "--frame-stride",
            str(int(args.frame_stride)),
            "--max-frames-per-scene",
            str(int(args.max_frames)),
            "--chunk-size",
            str(int(args.chunk_size)),
            "--first-prefix-count",
            "0",
            "--skip-assemble",
            "--skip-phase6-candidate",
            "--seed-source",
            "frame0_x1",
            "--x0-summary",
            str(args.x0_summary),
            "--allow-missing-x1-diagnostics",
            "--variant-id",
            f"{args.variant_id}_prepare",
            "--zip-name",
            "prepare_unused.zip",
        ]
        prepare_result = run_logged(prepare_cmd, log_path=log_dir / "prepare_x1_alltracker.log", env={**os.environ.copy(), "PYTHONUNBUFFERED": "1"})
        if int(prepare_result["returncode"]) != 0:
            raise RuntimeError({"failed_prepare_x1_alltracker": prepare_result})
        local_template_root = rel(output_root)
        args.x1_summary_template = f"{local_template_root}/x1seed_{{short}}_start{{frame_start:04d}}_f{{frame_count}}/baseline_x_gapadaptive_sam2/summary.json"
        args.frame0_birth_records_template = f"{local_template_root}/x1seed_{{short}}_start{{frame_start:04d}}_f{{frame_count}}/birth_bank/birth_records.json"
        args.alltracker_dir_template = f"{local_template_root}/alltracker_{{short}}_start{{frame_start:04d}}_f{{frame_count}}"

    procs = []
    for gpu in gpus:
        starts = [job["frame_start"] for job in assignments.get(gpu, [])]
        if not starts:
            continue
        procs.append(
            (
                gpu,
                launch_batch(
                    gpu=gpu,
                    starts=starts,
                    frame_count=int(args.chunk_size),
                    output_root=output_root,
                    summary_name=f"phase6_reuse_models_batch_gpu{gpu}_f32_summary.json",
                    log_path=log_dir / f"phase6_batch_gpu{gpu}_f32.log",
                    args=args,
                ),
            )
        )
    batch_results = [{"gpu": gpu, **collect_proc(proc)} for gpu, proc in procs]
    failed = [row for row in batch_results if int(row["returncode"]) != 0]
    if failed:
        raise RuntimeError({"failed_parallel_batches": failed})

    tail_results: list[dict[str, Any]] = []
    for tail in tail_specs:
        gpu = gpus[0]
        proc = launch_batch(
            gpu=gpu,
            starts=[tail["frame_start"]],
            frame_count=int(tail["frame_count"]),
            output_root=output_root,
            summary_name=f"phase6_reuse_models_batch_gpu{gpu}_f{tail['frame_count']}_summary.json",
            log_path=log_dir / f"phase6_batch_gpu{gpu}_f{tail['frame_count']}.log",
            args=args,
        )
        result = {"gpu": gpu, **collect_proc(proc)}
        tail_results.append(result)
        if int(result["returncode"]) != 0:
            raise RuntimeError({"failed_tail_batch": result})

    assembly_env = os.environ.copy()
    assembly_env["PYTHONUNBUFFERED"] = "1"
    assembly_cmd = [
        str(PYTHON),
        "tools/run_v105_fullscene_multichunk_repair.py",
        "--scenes",
        str(args.scene_id),
        "--output-root",
        rel(output_root),
        "--gpus",
        str(args.gpus),
        "--frame-stride",
        str(int(args.frame_stride)),
        "--max-frames-per-scene",
        str(int(args.max_frames)),
        "--chunk-size",
        str(int(args.chunk_size)),
        "--first-prefix-count",
        "0",
        "--skip-run",
        "--hardlink-assembled-labels",
        "--variant-id",
        str(args.variant_id),
        "--zip-name",
        str(args.zip_name),
        "--assembly-workers",
        str(int(args.assembly_workers)),
    ]
    if bool(args.posthoc_boundary_relabel):
        assembly_cmd.extend(
            [
                "--posthoc-boundary-relabel",
                "--posthoc-boundary-relabel-match-mode",
                str(args.posthoc_boundary_relabel_match_mode),
                "--posthoc-boundary-relabel-min-iou",
                str(float(args.posthoc_boundary_relabel_min_iou)),
                "--posthoc-boundary-relabel-min-intersection-pixels",
                str(int(args.posthoc_boundary_relabel_min_intersection_pixels)),
                "--posthoc-boundary-relabel-diagnostic-iou-threshold",
                str(float(args.posthoc_boundary_relabel_diagnostic_iou_threshold)),
                "--posthoc-boundary-relabel-workers",
                str(int(args.posthoc_boundary_relabel_workers)),
                "--posthoc-boundary-relabel-output-subdir",
                str(args.posthoc_boundary_relabel_output_subdir),
            ]
        )
        if str(args.posthoc_boundary_relabel_zip_name).strip():
            assembly_cmd.extend(["--posthoc-boundary-relabel-zip-name", str(args.posthoc_boundary_relabel_zip_name)])
        if bool(args.posthoc_boundary_relabel_skip_video):
            assembly_cmd.append("--posthoc-boundary-relabel-skip-video")
    assembly_result = run_logged(assembly_cmd, log_path=log_dir / "assembly.log", env=assembly_env)
    if int(assembly_result["returncode"]) != 0:
        raise RuntimeError({"failed_assembly": assembly_result})

    assembly_summary = output_root / "fullscene_multichunk_summary.json"
    assembly_summary_payload: dict[str, Any] = {}
    if assembly_summary.exists():
        assembly_summary_payload = json.loads(assembly_summary.read_text(encoding="utf-8"))
    zip_path = output_root / str(args.zip_name)
    total_wall = time.time() - started
    planned_frame_count = int(sum(int(spec["frame_count"]) for spec in specs))
    planned_chunk_count = int(len(specs))
    full_chunk_count = int(len(full_specs))
    tail_chunk_count = int(len(tail_specs))
    boundary = (
        "Unified wrapper for freshly prepared X1/AllTracker, persistent Phase6 batch, and optimized assembly; X0 summary is still an input artifact."
        if bool(args.prepare_x1_alltracker)
        else "Unified wrapper for persistent Phase6 batch plus optimized assembly; X1/AllTracker inputs are reused from frozen v105 roots."
    )
    summary = {
        "schema_version": "stream4d_v105_persistent_batch_pipeline_summary_v1",
        "scene_id": str(args.scene_id),
        "variant_id": str(args.variant_id),
        "output_root": rel(output_root),
        "frame_stride": int(args.frame_stride),
        "max_frames": int(args.max_frames),
        "chunk_size": int(args.chunk_size),
        "planned_frame_count": planned_frame_count,
        "planned_chunk_count": planned_chunk_count,
        "planned_full_chunk_count": full_chunk_count,
        "planned_tail_chunk_count": tail_chunk_count,
        "gpus": gpus,
        "planned_specs": specs,
        "duration_manifest": str(args.duration_manifest),
        "assignments": assignments,
        "prepare_x1_alltracker": bool(args.prepare_x1_alltracker),
        "prepare_result": prepare_result,
        "x1_summary_template": str(args.x1_summary_template),
        "frame0_birth_records_template": str(args.frame0_birth_records_template),
        "alltracker_dir_template": str(args.alltracker_dir_template),
        "parallel_batch_results": batch_results,
        "tail_batch_results": tail_results,
        "assembly_result": assembly_result,
        "posthoc_boundary_relabel_enabled": bool(args.posthoc_boundary_relabel),
        "posthoc_boundary_relabel_rows": assembly_summary_payload.get("posthoc_boundary_relabel_rows", []),
        "posthoc_boundary_relabel_all_ok": assembly_summary_payload.get("posthoc_boundary_relabel_all_ok", None),
        "pipeline_total_wall_sec": float(total_wall),
        "pipeline_sec_per_frame": float(total_wall / max(planned_frame_count, 1)),
        "pipeline_sec_per_chunk": float(total_wall / max(planned_chunk_count, 1)),
        "pipeline_sec_per_32frame_equiv_chunk": float(total_wall / max(planned_frame_count / 32.0, 1e-9)),
        "assembly_summary": rel(assembly_summary),
        "zip_path": rel(zip_path),
        "zip_exists": bool(zip_path.exists()),
        "zip_sha256": sha256_file(zip_path) if zip_path.exists() else "",
        "boundary": boundary,
    }
    summary_path = output_root / "persistent_batch_pipeline_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"summary": rel(summary_path), "zip": rel(zip_path), "pipeline_total_wall_sec": total_wall}, ensure_ascii=True), flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-id", default="scene0050_00")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--gpus", default="6,7")
    parser.add_argument("--frame-stride", type=int, default=5)
    parser.add_argument("--max-frames", type=int, default=931)
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--x0-summary", default=DEFAULT_X0_SUMMARY)
    parser.add_argument("--duration-manifest", default=DEFAULT_DURATION_MANIFEST)
    parser.add_argument("--x1-summary-template", default=DEFAULT_FROZEN_X1_SUMMARY_TEMPLATE)
    parser.add_argument("--frame0-birth-records-template", default=DEFAULT_FROZEN_BIRTH_RECORDS_TEMPLATE)
    parser.add_argument("--alltracker-dir-template", default=DEFAULT_FROZEN_ALLTRACKER_TEMPLATE)
    parser.add_argument(
        "--prepare-x1-alltracker",
        action="store_true",
        default=False,
        help="Freshly run frame0 X1 and AllTracker into --output-root before persistent Phase6 batches.",
    )
    parser.add_argument("--variant-id", default="P6_scene0050_persistent_batch_unified_blendlut_full931")
    parser.add_argument("--zip-name", default="scene0050_persistent_batch_unified_blendlut_full931.zip")
    parser.add_argument("--assembly-workers", type=int, default=1)
    parser.add_argument("--posthoc-boundary-relabel", action="store_true", default=False)
    parser.add_argument("--posthoc-boundary-relabel-match-mode", choices=("one_to_one", "split_friendly"), default="one_to_one")
    parser.add_argument("--posthoc-boundary-relabel-min-iou", type=float, default=0.05)
    parser.add_argument("--posthoc-boundary-relabel-min-intersection-pixels", type=int, default=512)
    parser.add_argument("--posthoc-boundary-relabel-diagnostic-iou-threshold", type=float, default=0.05)
    parser.add_argument("--posthoc-boundary-relabel-workers", type=int, default=8)
    parser.add_argument("--posthoc-boundary-relabel-skip-video", action="store_true", default=False)
    parser.add_argument("--posthoc-boundary-relabel-output-subdir", default="posthoc_boundary_relabel")
    parser.add_argument("--posthoc-boundary-relabel-zip-name", default="")
    parser.add_argument("--force", action="store_true", default=False)
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
