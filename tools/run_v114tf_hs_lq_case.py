#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = ROOT / "results/acl2_v114tf_semantic_internal_evidence_quality_memory_influence_control"
DEFAULT_IMAGE_ROOT = ROOT / "results/acl2_v113hs_horizonstream_semantic_aware_geometric_evidence_influence/kitti_generalizable"
DEFAULT_SEMANTIC_ROOT = ROOT / "results/acl2_v113hs_horizonstream_semantic_aware_geometric_evidence_influence/semantic_projection"
CONDA_SH = Path("/mnt/data/users/chengshun.wang/miniconda3/etc/profile.d/conda.sh")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one v114 HS-LQ HorizonStream case.")
    parser.add_argument("--action", required=True)
    parser.add_argument("--control", default="")
    parser.add_argument("--seq", required=True)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--results-root", default=str(DEFAULT_RESULTS))
    parser.add_argument("--image-root", default=str(DEFAULT_IMAGE_ROOT))
    parser.add_argument("--semantic-root", default=str(DEFAULT_SEMANTIC_ROOT))
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--conda-env", default="loger")
    parser.add_argument("--trace-enable", choices=["0", "1"], default="1")
    parser.add_argument("--action-audit-enable", choices=["0", "1"], default="1")
    parser.add_argument("--trace-gla-enable", choices=["0", "1"], default="0")
    parser.add_argument("--gq-layer-filter", default="")
    parser.add_argument("--la-layer-filter", default="")
    parser.add_argument("--chunk-block-num", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_root = Path(args.results_root).resolve()
    case_name = f"{args.output_prefix}_full_kitti_{args.seq}"
    if args.max_frames > 0:
        case_name = f"{args.output_prefix}_max{args.max_frames}_kitti_{args.seq}"
    output_root = results_root / "outputs" / case_name
    audit_root = results_root / "diagnostics" / case_name
    log_path = results_root / "logs" / f"{case_name}.log"
    manifest_path = audit_root / "run_manifest.json"

    output_root.parent.mkdir(parents=True, exist_ok=True)
    audit_root.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    config_path = ROOT / "third_party/HorizonStream/configs/horizonstream_infer.yaml"
    original_config_path = config_path
    if args.chunk_block_num > 0:
        cfg = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        cfg.setdefault("model", {}).setdefault("horizonstream_cfg", {}).setdefault("agg_regator_cfg", {})[
            "chunk_block_num"
        ] = int(args.chunk_block_num)
        config_path = audit_root / f"derived_chunk_block_num_{int(args.chunk_block_num)}.yaml"
        config_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    env_parts = [
        "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True",
        f"HS_V113_ACTION={args.action}",
        f"HS_V113_SEMANTIC_ROOT={Path(args.semantic_root).resolve()}",
        f"CUDA_VISIBLE_DEVICES={args.gpu}",
    ]
    if args.action_audit_enable == "1":
        env_parts.insert(3, f"HS_V114_ACTION_AUDIT_ROOT={audit_root}")
    if args.trace_enable == "1":
        env_parts.insert(4 if args.action_audit_enable == "1" else 3, "HS_V113_TRACE_ENABLE=1")
        env_parts.insert(5 if args.action_audit_enable == "1" else 4, f"HS_V113_TRACE_ROOT={audit_root}")
        env_parts.insert(6 if args.action_audit_enable == "1" else 5, "HS_V113_TRACE_LOCAL_ENABLE=0")
        env_parts.insert(7 if args.action_audit_enable == "1" else 6, f"HS_V113_TRACE_GLA_ENABLE={args.trace_gla_enable}")
        env_parts.insert(8 if args.action_audit_enable == "1" else 7, "HS_V113_TRACE_MRT_ENABLE=1")
    else:
        env_parts.insert(3 if args.action_audit_enable == "0" else 4, "HS_V113_TRACE_ENABLE=0")
    if args.control:
        env_parts.insert(2, f"HS_V113_CONTROL={args.control}")
    if args.gq_layer_filter:
        env_parts.insert(3, f"HS_V115_GQ_LAYER_FILTER={args.gq_layer_filter}")
    if args.la_layer_filter:
        env_parts.insert(3, f"HS_V116_LA_LAYER_FILTER={args.la_layer_filter}")
    max_frames = f" --max-frames {int(args.max_frames)}" if args.max_frames > 0 else ""
    command = (
        f"source {CONDA_SH} && conda activate {args.conda_env} && "
        f"cd {ROOT / 'third_party/HorizonStream'} && "
        + " ".join(env_parts)
        + f" python run_pipeline.py --config {config_path} "
        f"--img-path {Path(args.image_root).resolve()} --seq-list {args.seq} --camera 02 "
        "--checkpoint checkpoints/HorizonStream.pt "
        f"--output-root {output_root}{max_frames} "
        "--no-camera-preprocess --offload-outputs-to-cpu "
        "--no-save-videos --no-save-points --no-save-images --no-save-depth --no-save-depth-conf "
        "--no-mask-sky --no-point-mask-sky --no-loop --eval-pose-variants main"
    )
    manifest = {
        "schema": "acl2_v114tf_hs_lq_case_manifest_v1",
        "action": args.action,
        "control": args.control,
        "seq": args.seq,
        "gpu": args.gpu,
        "output_prefix": args.output_prefix,
        "output_root": str(output_root),
        "audit_root": str(audit_root),
        "log_path": str(log_path),
        "trace_enable": args.trace_enable,
        "action_audit_enable": args.action_audit_enable,
        "trace_gla_enable": args.trace_gla_enable,
        "gq_layer_filter": args.gq_layer_filter,
        "la_layer_filter": args.la_layer_filter,
        "chunk_block_num": int(args.chunk_block_num),
        "config_path": str(config_path),
        "original_config_path": str(original_config_path),
        "command": command,
        "start_time_unix": time.time(),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    proc: subprocess.Popen[str] | None = None
    rc = 1
    interrupted = False
    try:
        with log_path.open("w", encoding="utf-8") as log:
            log.write("# v114 HS-LQ case command\n")
            log.write(command + "\n\n")
            log.flush()
            proc = subprocess.Popen(
                ["bash", "-lc", command],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                start_new_session=True,
            )
            assert proc.stdout is not None
            for line in proc.stdout:
                sys.stdout.write(line)
                log.write(line)
            rc = proc.wait()
    except KeyboardInterrupt:
        interrupted = True
        rc = 130
        if proc is not None and proc.poll() is None:
            try:
                os.killpg(proc.pid, signal.SIGINT)
            except ProcessLookupError:
                pass
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(proc.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                proc.wait(timeout=30)

    manifest["end_time_unix"] = time.time()
    manifest["returncode"] = int(rc)
    if interrupted:
        manifest["interrupted"] = True
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"case": case_name, "returncode": int(rc), "manifest": str(manifest_path)}, sort_keys=True))
    raise SystemExit(rc)


if __name__ == "__main__":
    main()
