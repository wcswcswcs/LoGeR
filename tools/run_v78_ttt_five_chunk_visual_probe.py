#!/usr/bin/env python3
"""Run v78 TTT output-separated visual probes on a multi-chunk window.

This wrapper does not introduce a new TTT method. It only reproduces the
already validated Phase4 dump path on selected bad/reference chunks:

1. T0_NATIVE_PCA: save TTT operator/update/final PCA features.
2. T0_TTT_WRITE_PROBE_DELTA: save projected TTT post-zp delta maps.
3. visualize_v78_phase4_ttt_output_separated.py: build auditable panels.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import shlex
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONDA = Path("/mnt/data/users/chengshun.wang/miniconda3/bin/conda")
DEFAULT_PY = Path("/mnt/data/users/chengshun.wang/miniconda3/envs/loger/bin/python")
DEFAULT_CHECKPOINT = REPO_ROOT / "ckpts/LoGeR/latest.pt"
DEFAULT_CONFIG = REPO_ROOT / "ckpts/LoGeR/original_config.yaml"
DEFAULT_DATA = Path("/mnt/data/users/chengshun.wang/data/kitti_odometry/dataset")


def _parse_ints(text: str) -> List[int]:
    out: List[int] = []
    for part in str(text or "").replace(";", ",").split(","):
        part = part.strip()
        if part:
            out.append(int(part))
    return out


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _chunk_window(chunk: int, chunk_size: int, chunk_overlap: int) -> Dict[str, int]:
    stride = int(chunk_size) - int(chunk_overlap)
    start = int(chunk) * stride
    end = start + int(chunk_size)
    return {
        "chunk": int(chunk),
        "start_frame": int(start),
        "end_frame": int(end),
        "mid_frame": int(start + chunk_size // 2),
        "last_frame": int(end - 1),
    }


def _stage_c_masklet(stage_c_root: Path, chunk: int) -> Path:
    matches = sorted(stage_c_root.glob(f"chunk_{int(chunk):03d}_*/masklet.pt"))
    if not matches:
        raise FileNotFoundError(f"missing stage-c masklet for chunk {chunk}: {stage_c_root}/chunk_{chunk:03d}_*/masklet.pt")
    return matches[0]


def _run_job(job: Dict[str, Any]) -> Dict[str, Any]:
    env = os.environ.copy()
    env["PYTHONPATH"] = "."
    env["CUDA_VISIBLE_DEVICES"] = str(job["gpu"])
    if str(job.get("cuda_alloc_conf") or "").strip():
        env["PYTORCH_CUDA_ALLOC_CONF"] = str(job["cuda_alloc_conf"]).strip()
    out_dir = Path(job["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "run.log"
    start = time.time()
    with log_path.open("w", encoding="utf-8") as handle:
        proc = subprocess.run(
            job["cmd"],
            cwd=str(REPO_ROOT),
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    job["returncode"] = int(proc.returncode)
    job["duration_sec"] = float(time.time() - start)
    job["run_log"] = str(log_path)
    return job


def _build_pipeline_cmd(
    args: argparse.Namespace,
    *,
    chunk: int,
    case: str,
    out_dir: Path,
) -> List[str]:
    window = _chunk_window(chunk, args.chunk_size, args.chunk_overlap)
    cmd = [
        str(args.conda),
        "run",
        "--no-capture-output",
        "-n",
        args.conda_env,
        "python",
        "run_pipeline_abc_v2.py",
        "--input",
        str(args.input_dir),
        "--output_video",
        "",
        "--output_txt",
        str(out_dir / f"{args.seq}.txt"),
        "--checkpoint",
        str(args.checkpoint),
        "--config",
        str(args.config),
        "--chunk_size",
        str(args.chunk_size),
        "--chunk_overlap",
        str(args.chunk_overlap),
        "--start_frame",
        str(window["start_frame"]),
        "--end_frame",
        str(window["end_frame"]),
        "--global_chunk_offset",
        str(chunk),
        "--device",
        "cuda",
        "--semantic_prior_mode",
        "spg_v2",
        "--stage_c_cache_mode",
        "read",
        "--stage_c_cache_dir",
        str(args.stage_c_cache_dir),
        "--stage_c_cache_require_hit",
        "1",
        "--read_path",
        "none",
        "--read_cue_source",
        "dyn",
        "--fast_cue_eval",
        "1",
        "--empty_cuda_cache_each_chunk",
        "1",
        "--hybrid_debug_jsonl",
        str(out_dir / "hmc_state_hash.jsonl"),
    ]
    if case == "T0_NATIVE_PCA":
        cmd.extend(
            [
                "--hybrid_memory_mode",
                "read_path_only",
                "--hmc_commit_mode",
                "controlled",
                "--enable_frame_read_control",
                "0",
                "--v68_export_full_pca_debug",
                "1",
                "--v68_pca_max_feature_dim",
                str(args.v68_pca_max_feature_dim),
                "--v68_layer_pca_feature_dir",
                str(out_dir / "pca_features"),
                "--v68_pca_taps",
                "ttt_operator_output,ttt_update_term,ttt_final_output",
                "--v68_pca_layers",
                str(args.v68_pca_layers),
            ]
        )
    elif case == "T0_TTT_WRITE_PROBE_DELTA":
        cmd.extend(
            [
                "--hybrid_memory_mode",
                "ttt_write_only",
                "--hmc_commit_mode",
                "probe_ttt_write",
                "--ttt_spatial_post_delta_map_dump_dir",
                str(out_dir / "ttt_spatial_post_delta_maps"),
                "--ttt_spatial_post_delta_map_dump_dtype",
                str(args.ttt_spatial_post_delta_map_dump_dtype),
            ]
        )
    else:
        raise ValueError(f"unknown case: {case}")
    return cmd


def _run_visualize(args: argparse.Namespace, *, chunk: int, out_root: Path) -> Dict[str, Any]:
    window = _chunk_window(chunk, args.chunk_size, args.chunk_overlap)
    native_dir = out_root / f"chunk{chunk:03d}" / "T0_NATIVE_PCA"
    probe_dir = out_root / f"chunk{chunk:03d}" / "T0_TTT_WRITE_PROBE_DELTA"
    pca_pt = native_dir / "pca_features" / f"chunk_{chunk:03d}.pt"
    post_delta_pt = probe_dir / "ttt_spatial_post_delta_maps" / f"chunk_{chunk:03d}_ttt_spatial_post_delta_map.pt"
    stage_c_masklet = _stage_c_masklet(args.stage_c_cache_dir, chunk)
    visual_dir = out_root / "visual_output_separated" / f"chunk{chunk:03d}"
    frames = f"{window['start_frame']},{window['mid_frame']},{window['last_frame']}"
    cmd = [
        str(args.python),
        "tools/visualize_v78_phase4_ttt_output_separated.py",
        "--pca-pt",
        str(pca_pt),
        "--stage-c-masklet",
        str(stage_c_masklet),
        "--post-delta-pt",
        str(post_delta_pt),
        "--rgb-dir",
        str(args.input_dir),
        "--out-dir",
        str(visual_dir),
        "--frames",
        frames,
        "--operator-layers",
        str(args.operator_layers),
        "--single-layer",
        str(args.single_layer),
        "--panel-width",
        str(args.panel_width),
    ]
    log_path = visual_dir / "visualize.log"
    visual_dir.mkdir(parents=True, exist_ok=True)
    start = time.time()
    with log_path.open("w", encoding="utf-8") as handle:
        proc = subprocess.run(cmd, cwd=str(REPO_ROOT), stdout=handle, stderr=subprocess.STDOUT, check=False)
    audit_path = visual_dir / "visual_integrity_audit.json"
    audit: Dict[str, Any] = {}
    if audit_path.exists():
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
    return {
        "chunk": int(chunk),
        "frames": frames,
        "cmd": cmd,
        "cmd_shell": shlex.join(cmd),
        "returncode": int(proc.returncode),
        "duration_sec": float(time.time() - start),
        "visual_dir": str(visual_dir),
        "visualize_log": str(log_path),
        "pca_pt": str(pca_pt),
        "post_delta_pt": str(post_delta_pt),
        "stage_c_masklet": str(stage_c_masklet),
        "visual_audit": str(audit_path),
        "visual_gate_pass": bool(audit.get("gate_pass")),
        "audit_payload": audit,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seq", default="02")
    parser.add_argument("--chunks", default="64,65,66,67,68")
    parser.add_argument("--gpus", default="0,1,2,3,4,5")
    parser.add_argument("--out-root", type=Path, default=None)
    parser.add_argument("--conda", type=Path, default=DEFAULT_CONDA)
    parser.add_argument("--python", type=Path, default=DEFAULT_PY)
    parser.add_argument("--conda-env", default="loger")
    parser.add_argument("--input-dir", type=Path, default=None)
    parser.add_argument("--gt", type=Path, default=None)
    parser.add_argument("--stage-c-cache-dir", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument("--chunk-overlap", type=int, default=3)
    parser.add_argument("--v68-pca-layers", default="6,14,18")
    parser.add_argument("--v68-pca-max-feature-dim", type=int, default=8)
    parser.add_argument("--operator-layers", default="6,14,18")
    parser.add_argument("--single-layer", type=int, default=18)
    parser.add_argument("--panel-width", type=int, default=420)
    parser.add_argument("--ttt-spatial-post-delta-map-dump-dtype", default="float16")
    parser.add_argument("--cuda-alloc-conf", default="expandable_segments:True")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--no-run", action="store_true")
    parser.add_argument("--no-visualize", action="store_true")
    args = parser.parse_args()

    seq = str(args.seq).zfill(2)
    args.seq = seq
    if args.input_dir is None:
        args.input_dir = DEFAULT_DATA / "sequences" / seq / "image_2"
    if args.gt is None:
        args.gt = DEFAULT_DATA / "poses" / f"{seq}.txt"
    if args.stage_c_cache_dir is None:
        args.stage_c_cache_dir = Path(f"results/kitti_preprocess/{seq}/stage_c_cache_semantic_chunks")
    if args.out_root is None:
        args.out_root = Path(
            f"results/kitti{seq}_hmc_v2/acl2_v78tf_pca_grounded_memory_control/"
            "report_final/phase4_ttt_five_chunk_regime_visual_probe_v1"
        )

    chunks = _parse_ints(args.chunks)
    gpus = _parse_ints(args.gpus)
    if not chunks:
        raise ValueError("--chunks is empty")
    if not gpus:
        raise ValueError("--gpus is empty")
    if not args.input_dir.exists():
        raise FileNotFoundError(args.input_dir)
    if not args.gt.exists():
        raise FileNotFoundError(args.gt)
    if not args.stage_c_cache_dir.exists():
        raise FileNotFoundError(args.stage_c_cache_dir)

    out_root = args.out_root
    out_root.mkdir(parents=True, exist_ok=True)
    cases = ["T0_NATIVE_PCA", "T0_TTT_WRITE_PROBE_DELTA"]
    jobs: List[Dict[str, Any]] = []
    gpu_cursor = 0
    for chunk in chunks:
        _stage_c_masklet(args.stage_c_cache_dir, chunk)
        for case in cases:
            out_dir = out_root / f"chunk{chunk:03d}" / case
            expected = (
                out_dir / "pca_features" / f"chunk_{chunk:03d}.pt"
                if case == "T0_NATIVE_PCA"
                else out_dir / "ttt_spatial_post_delta_maps" / f"chunk_{chunk:03d}_ttt_spatial_post_delta_map.pt"
            )
            skipped = bool(args.skip_existing and expected.exists())
            cmd = _build_pipeline_cmd(args, chunk=chunk, case=case, out_dir=out_dir)
            jobs.append(
                {
                    "chunk": int(chunk),
                    "case": case,
                    "gpu": int(gpus[gpu_cursor % len(gpus)]),
                    "out_dir": str(out_dir),
                    "cmd": cmd,
                    "cmd_shell": shlex.join(cmd),
                    "expected": str(expected),
                    "skipped": skipped,
                    "returncode": 0 if skipped else None,
                    "cuda_alloc_conf": str(args.cuda_alloc_conf),
                }
            )
            gpu_cursor += 1

    manifest_path = out_root / "ttt_five_chunk_visual_probe_run_manifest.json"
    manifest_path.write_text(
        json.dumps(_jsonable({"args": vars(args), "jobs": jobs}), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"planned_jobs={len(jobs)} manifest={manifest_path}")
    if args.no_run:
        return

    completed: List[Dict[str, Any]] = [job for job in jobs if job["skipped"]]
    run_jobs = [job for job in jobs if not job["skipped"]]
    jobs_by_gpu: Dict[int, List[Dict[str, Any]]] = {int(gpu): [] for gpu in gpus}
    for job in run_jobs:
        jobs_by_gpu[int(job["gpu"])].append(job)

    def _run_gpu_queue(gpu: int, queue: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [_run_job(dict(job)) for job in queue]

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        futures = {
            pool.submit(_run_gpu_queue, gpu, queue): gpu
            for gpu, queue in jobs_by_gpu.items()
            if queue
        }
        for future in concurrent.futures.as_completed(futures):
            completed.extend(future.result())

    completed = sorted(completed, key=lambda row: (int(row["chunk"]), str(row["case"])))
    run_summary_path = out_root / "ttt_five_chunk_visual_probe_run_summary.json"
    run_ok = all(int(row.get("returncode") or 0) == 0 and Path(str(row["expected"])).exists() for row in completed)
    visual_rows: List[Dict[str, Any]] = []
    if run_ok and not args.no_visualize:
        for chunk in chunks:
            visual_rows.append(_run_visualize(args, chunk=chunk, out_root=out_root))

    visual_ok = bool(visual_rows) and all(int(row["returncode"]) == 0 and bool(row.get("visual_gate_pass")) for row in visual_rows)
    artifact_hashes: Dict[str, str] = {}
    for path in sorted(out_root.glob("**/*")):
        if path.is_file() and path.name in {
            "visual_integrity_audit.json",
            "visual_artifact_manifest.csv",
            "visual_review.csv",
            "visual_insight.md",
            "write_role_mass.csv",
        }:
            artifact_hashes[str(path)] = _sha256(path)
    summary = {
        "schema": "acl2_v78_ttt_five_chunk_visual_probe_summary_v1",
        "seq": seq,
        "chunks": chunks,
        "run_manifest": str(manifest_path),
        "jobs": completed,
        "all_pipeline_jobs_ok": bool(run_ok),
        "visualizations": visual_rows,
        "all_visual_gates_pass": bool(visual_ok),
        "method_gate_claimed": False,
        "diagnostic_only": True,
        "artifact_hashes": artifact_hashes,
    }
    run_summary_path.write_text(json.dumps(_jsonable(summary), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(_jsonable({
        "out_root": out_root,
        "all_pipeline_jobs_ok": run_ok,
        "all_visual_gates_pass": visual_ok,
        "run_summary": run_summary_path,
        "diagnostic_only": True,
        "method_gate_claimed": False,
    }), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
