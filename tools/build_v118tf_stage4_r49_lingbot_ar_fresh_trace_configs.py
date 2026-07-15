#!/usr/bin/env python3
"""Build fresh LingBot no-action trace configs for ACL2 v118 Stage4-R49.

R49 prepares the missing upstream evidence for the R47 abstention-policy
candidate.  It runs no semantic action: the method is the default FlashInfer
streaming LingBot path with v118 provenance tracing enabled by environment
variables at runtime.
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "results/acl2_v118tf_operation_specific_semantic_carrier_calibration"


def parse_seq_env(name: str, default: str) -> tuple[str, ...]:
    seqs = tuple(part.strip().zfill(2) for part in os.environ.get(name, default).replace(";", ",").split(",") if part.strip())
    return seqs or tuple(part.strip().zfill(2) for part in default.split(",") if part.strip())


STAGE_TAG = os.environ.get("ACL2_V118_FRESH_TRACE_TAG", "r49").strip().lower() or "r49"
STAGE = RESULT_ROOT / os.environ.get("ACL2_V118_FRESH_TRACE_STAGE_SLUG", "stage4_r49_lingbot_ar_fresh_trace_baseline")
CONFIG_DIR = STAGE / "configs"
RUNTIME = STAGE / "runtime_full"
WORKSPACE = STAGE / "workspace"
BENCH = ROOT / "third_party/lingbot-map/benchmark"
CONDA = Path("/mnt/data/users/chengshun.wang/miniconda3/bin/conda")
PYTHONPATH = f"{ROOT / 'third_party/lingbot-map'}:{ROOT / 'third_party/lingbot-map/benchmark'}"
FRESH_SEQS = parse_seq_env("ACL2_V118_FRESH_TRACE_SEQS", "04,03")
DATASET_PREFIX = os.environ.get("ACL2_V118_FRESH_DATASET_PREFIX", "kitti_v118_r49_fresh_seq")
METHOD = os.environ.get("ACL2_V118_FRESH_TRACE_METHOD", f"lingbot_map_stream_flashinfer_v118_{STAGE_TAG}_fresh_trace")
CONFIG_NAME = os.environ.get(
    "ACL2_V118_FRESH_TRACE_CONFIG_NAME",
    f"kitti_lingbot_flashinfer_{STAGE_TAG}_fresh_trace_{'_'.join(FRESH_SEQS)}.yaml",
)


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


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
    path.write_text(json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def dataset_name(seq: str) -> str:
    return f"{DATASET_PREFIX}{seq}"


def dataset_yaml(seq: str) -> str:
    return "\n".join(
        [
            "dataset: kitti",
            f"raw_data_root: {ROOT / 'data/kitti/dataset'}",
            "_target_size: [504, 280]",
            f'_sequences: ["{seq}"]',
            "",
        ]
    )


def method_yaml() -> str:
    return "\n".join(
        [
            "model: lingbot_map",
            "env: loger",
            f"_checkpoint: {ROOT / 'third_party/lingbot-map/checkpoints/lingbot-map-long.pt'}",
            "_device: cuda",
            "_use_amp: true",
            "_use_sdpa: false",
            "_image_size: 518",
            "_patch_size: 14",
            "_enable_3d_rope: true",
            "_num_scale_frames: 8",
            "_max_frame_num: 1024",
            "_kv_cache_sliding_window: 64",
            "_kv_cache_scale_frames: 8",
            "_auto_keyframe_threshold: 320",
            "_area_budget: 255000",
            "_align: 14",
            "_mode: streaming",
            "_keyframe_interval: auto",
            "",
        ]
    )


def main_config() -> str:
    return "\n".join(
        [
            f"workspace: {WORKSPACE}",
            "",
            "evaluation:",
            "  traj:",
            "    enable: true",
            "    vis: true",
            "  auc:",
            "    enable: false",
            "  depth:",
            "    enable: false",
            "  points:",
            "    enable: false",
            "",
            "datasets:",
            *[f"  - {dataset_name(seq)}" for seq in FRESH_SEQS],
            "",
            "methods:",
            f"  - {METHOD}",
            "",
        ]
    )


def env_prefix(seq: str, gpu: int) -> str:
    trace = RUNTIME / f"seq{seq}_flashinfer_trace.jsonl"
    return (
        f"PATH=/mnt/data/users/chengshun.wang/miniconda3/bin:$PATH "
        f"PYTHONPATH={PYTHONPATH} "
        f"CUDA_VISIBLE_DEVICES={gpu} "
        f"ACL2_V118_LB_FI_PROVENANCE_FILE={trace.resolve()} "
        f"ACL2_V118_LB_PROVENANCE_SEQ={seq}"
    )


def main() -> int:
    (CONFIG_DIR / "datasets").mkdir(parents=True, exist_ok=True)
    (CONFIG_DIR / "methods").mkdir(parents=True, exist_ok=True)
    (RUNTIME / "logs").mkdir(parents=True, exist_ok=True)
    for seq in FRESH_SEQS:
        write_text(CONFIG_DIR / "datasets" / f"{dataset_name(seq)}.yaml", dataset_yaml(seq))
    write_text(CONFIG_DIR / "methods" / f"{METHOD}.yaml", method_yaml())
    config = CONFIG_DIR / CONFIG_NAME
    write_text(config, main_config())

    rows: list[dict[str, Any]] = []
    common = f"PATH=/mnt/data/users/chengshun.wang/miniconda3/bin:$PATH PYTHONPATH={PYTHONPATH}"
    seq_tag = "_".join(FRESH_SEQS)
    prepare_log = RUNTIME / f"logs/prepare_fresh_{seq_tag}.log"
    rows.append(
        {
            "schema": "acl2_v118tf_stage4_r49_fresh_trace_run_manifest_row_v1",
            "phase": "prepare",
            "seq": ",".join(FRESH_SEQS),
            "dataset": ",".join(dataset_name(seq) for seq in FRESH_SEQS),
            "method": METHOD,
            "gpu": "",
            "cwd": str(BENCH),
            "config": str(config.resolve()),
            "trace": "",
            "log": rel(prepare_log),
            "command": f"{common} {CONDA} run -n loger --no-capture-output python prepare.py --config {config.resolve()} --force > {prepare_log} 2>&1",
        }
    )
    for gpu, seq in enumerate(FRESH_SEQS):
        trace = RUNTIME / f"seq{seq}_flashinfer_trace.jsonl"
        log = RUNTIME / f"logs/run_seq{seq}_gpu{gpu}.log"
        rows.append(
            {
                "schema": "acl2_v118tf_stage4_r49_fresh_trace_run_manifest_row_v1",
                "phase": "run_worker",
                "seq": seq,
                "dataset": dataset_name(seq),
                "method": METHOD,
                "gpu": gpu,
                "cwd": str(BENCH),
                "config": str(config.resolve()),
                "trace": rel(trace),
                "log": rel(log),
                "command": (
                    f"{env_prefix(seq, gpu)} {CONDA} run -n loger --no-capture-output "
                    f"python run_worker.py --config {config.resolve()} --method {METHOD} "
                    f"--dataset {dataset_name(seq)} --scene {seq} --force > {log} 2>&1"
                ),
            }
        )
    eval_log = RUNTIME / f"logs/evaluate_fresh_{seq_tag}.log"
    rows.append(
        {
            "schema": "acl2_v118tf_stage4_r49_fresh_trace_run_manifest_row_v1",
            "phase": "evaluate",
            "seq": ",".join(FRESH_SEQS),
            "dataset": ",".join(dataset_name(seq) for seq in FRESH_SEQS),
            "method": METHOD,
            "gpu": "",
            "cwd": str(BENCH),
            "config": str(config.resolve()),
            "trace": "",
            "log": rel(eval_log),
            "command": f"{common} {CONDA} run -n loger --no-capture-output python evaluate.py --config {config.resolve()} --force > {eval_log} 2>&1",
        }
    )
    write_csv(STAGE / "run_manifest.csv", rows)

    summary = {
        "schema": "acl2_v118tf_stage4_r49_fresh_trace_config_summary_v1",
        "stage4_r49_decision": "FRESH_TRACE_CONFIGS_READY_NOT_RUN",
        "global_goal_achieved": False,
        "sequences": list(FRESH_SEQS),
        "method": METHOD,
        "workspace": rel(WORKSPACE),
        "config": rel(config),
        "run_manifest": rel(STAGE / "run_manifest.csv"),
        "trace_outputs": {seq: rel(RUNTIME / f"seq{seq}_flashinfer_trace.jsonl") for seq in FRESH_SEQS},
        "boundary": "R49 configs run no semantic action. They only create fresh default LingBot geometry and internal FlashInfer trace evidence for later pre-registered R47 validation.",
    }
    write_json(STAGE / "config_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
