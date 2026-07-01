#!/usr/bin/env python3
"""Generate ACL2 v105-TF LingBot Stage 4 action-pilot configs."""

from __future__ import annotations

import csv
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "results/acl2_v105tf_dualtrack_lingbot_loger_evidence_eligibility"
CONFIG_ROOT = RESULT_ROOT / "configs"
METHOD_DIR = CONFIG_ROOT / "methods"
STAGE3 = RESULT_ROOT / "stage3_lingbot_oracle"
STAGE4 = RESULT_ROOT / "stage4_lingbot_action_pilot_or_blocked"
STAGE4_HEAD = RESULT_ROOT / "stage4_lingbot_headlocal_trace"
WORKSPACE = STAGE4 / "workspace"
RAW_TRACE = STAGE4 / "raw_trace"
RAW_ACTION = STAGE4 / "raw_action"
BENCHMARK = ROOT / "third_party/lingbot-map/benchmark"
CONDA = Path("/mnt/data/users/chengshun.wang/miniconda3/bin/conda")

SEQUENCES = ["00", "02"]
POLICY = "semantic_reject_unreliable_attention_frac_ge_0.07418_AND_local_window_context_attention_frac_ge_0.7209"
SEMANTIC_THR = 0.07418
LOCAL_WINDOW_THR = 0.7209
SCALE_FRAMES = 8


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


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


def as_float(row: dict[str, str], key: str) -> float:
    return float(row.get(key, 0.0) or 0.0)


def as_int(row: dict[str, str], key: str) -> int:
    return int(float(row.get(key, 0) or 0))


def by_seq(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    out: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        out[str(row["seq"])].append(row)
    for seq_rows in out.values():
        seq_rows.sort(key=lambda row: as_int(row, "sample_position"))
    return out


def selected_indices(rows: list[dict[str, str]]) -> list[int]:
    return sorted({as_int(row, "sample_position") for row in rows if as_int(row, "sample_position") >= SCALE_FRAMES})


def intersect_indices(lhs: list[int], rhs: list[int]) -> list[int]:
    return sorted(set(lhs).intersection(rhs))


def action_indices(frame_rows: list[dict[str, str]]) -> dict[str, dict[str, list[int]]]:
    selected_rows = read_csv(STAGE3 / "stage3_sweep_passing_policy_selected_rows.csv")
    selected_by_seq = by_seq(selected_rows)
    safety_rows_path = STAGE4 / "semantic_safety_filter_selected_rows.csv"
    safety_by_seq = by_seq(read_csv(safety_rows_path)) if safety_rows_path.exists() else {}
    headlocal_rows_path = STAGE4_HEAD / "headlocal_relaxed_selected_rows.csv"
    headlocal_by_seq = by_seq(read_csv(headlocal_rows_path)) if headlocal_rows_path.exists() else {}
    frame_by_seq = by_seq(frame_rows)
    out: dict[str, dict[str, list[int]]] = {seq: {} for seq in SEQUENCES}
    rng = random.Random(1054)

    for seq in SEQUENCES:
        seq_rows = frame_by_seq[seq]
        semgeom = selected_indices(selected_by_seq.get(seq, []))
        out[seq]["semantic_geometry_write_filter"] = semgeom
        out[seq]["semantic_geometry_context_only_demote"] = semgeom
        strong_safety = selected_indices(safety_by_seq.get(seq, []))
        if strong_safety:
            out[seq]["semantic_safety_strong_context_only_demote"] = strong_safety
            out[seq]["semantic_safety_anchor_only_demote"] = strong_safety
        headlocal = selected_indices(headlocal_by_seq.get(seq, []))
        if headlocal:
            out[seq]["semantic_headlocal_relaxed_context_only_demote"] = headlocal
            if strong_safety:
                headlocal_safety = intersect_indices(headlocal, strong_safety)
                if headlocal_safety:
                    out[seq]["semantic_headlocal_safety_context_only_demote"] = headlocal_safety
        out[seq]["semantic_only_reject_write_filter"] = selected_indices(
            [row for row in seq_rows if as_float(row, "semantic_reject_unreliable_attention_frac") >= SEMANTIC_THR]
        )
        out[seq]["geometry_only_local_window_write_filter"] = selected_indices(
            [row for row in seq_rows if as_float(row, "local_window_context_attention_frac") >= LOCAL_WINDOW_THR]
        )

        candidates = [as_int(row, "sample_position") for row in seq_rows if as_int(row, "sample_position") >= SCALE_FRAMES]
        out[seq]["same_count_random_write_filter"] = sorted(rng.sample(candidates, k=min(len(semgeom), len(candidates))))

        shifted_rows: list[dict[str, str]] = []
        for idx, row in enumerate(seq_rows):
            shifted = dict(row)
            src = seq_rows[(idx + 1) % len(seq_rows)]
            shifted["semantic_reject_unreliable_attention_frac"] = src["semantic_reject_unreliable_attention_frac"]
            shifted_rows.append(shifted)
        out[seq]["semantic_shuffle_write_filter"] = selected_indices(
            [
                row
                for row in shifted_rows
                if as_float(row, "semantic_reject_unreliable_attention_frac") >= SEMANTIC_THR
                and as_float(row, "local_window_context_attention_frac") >= LOCAL_WINDOW_THR
            ]
        )

        out[seq]["context_role_rotation_write_filter"] = selected_indices(
            [
                row
                for row in seq_rows
                if as_float(row, "semantic_reject_unreliable_attention_frac") >= SEMANTIC_THR
                and as_float(row, "scale_reference_context_attention_frac") >= LOCAL_WINDOW_THR
            ]
        )
    return out


def method_yaml(method: str, indices: list[int], label: str) -> str:
    if label.endswith("_context_only_demote"):
        action_mode = "context_only_special"
    elif label.endswith("_anchor_only_demote"):
        action_mode = "anchor_special_only"
    else:
        action_mode = "force_non_keyframe"
    return "\n".join(
        [
            "model: lingbot_map",
            "env: loger",
            f"_checkpoint: {ROOT / 'third_party/lingbot-map/checkpoints/lingbot-map-long.pt'}",
            "_device: cuda",
            "_use_amp: true",
            "_use_sdpa: true",
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
            f"_force_non_keyframe_indices: {json.dumps(indices)}",
            f"_stage4_action_label: {label}",
            f"_stage4_action_mode: {action_mode}",
            "",
        ]
    )


def base_yaml(dataset: str, method: str) -> str:
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
            f"  - {dataset}",
            "",
            "methods:",
            f"  - {method}",
            "",
        ]
    )


def command_prefix(gpu: int) -> str:
    return (
        f"PATH={CONDA.parent}:$PATH "
        f"PYTHONPATH={ROOT / 'third_party/lingbot-map'}:{ROOT / 'third_party/lingbot-map/benchmark'} "
        f"CUDA_VISIBLE_DEVICES={gpu}"
    )


def build() -> dict[str, Any]:
    STAGE4.mkdir(parents=True, exist_ok=True)
    RAW_TRACE.mkdir(parents=True, exist_ok=True)
    RAW_ACTION.mkdir(parents=True, exist_ok=True)
    frame_rows = read_csv(STAGE3 / "frame_semantic_geometry_rows.csv")
    indices_by_seq = action_indices(frame_rows)

    manifest: list[dict[str, Any]] = []
    action_rows: list[dict[str, Any]] = []
    for seq in SEQUENCES:
        dataset = f"kitti_v105_seq{seq}_trace32"
        gpu = 0 if seq == "00" else 2
        for action_label, indices in sorted(indices_by_seq[seq].items()):
            method = f"lingbot_map_stage4_{action_label}_seq{seq}"
            config = CONFIG_ROOT / f"kitti_lingbot_stage4_{action_label}_seq{seq}_trace32.yaml"
            method_path = METHOD_DIR / f"{method}.yaml"
            trace_file = RAW_TRACE / f"{dataset}_{seq}_{method}.jsonl"
            action_file = RAW_ACTION / f"{dataset}_{seq}_{method}.jsonl"
            run_name = f"kitti_lingbot_stage4_{action_label}_seq{seq}_trace32"

            method_path.write_text(method_yaml(method, indices, action_label), encoding="utf-8")
            config.write_text(base_yaml(dataset, method), encoding="utf-8")
            action_rows.append(
                {
                    "schema": "acl2_v105tf_lingbot_stage4_action_config_v1",
                    "seq": seq,
                    "dataset": dataset,
                    "method": method,
                    "action_label": action_label,
                    "force_non_keyframe_indices": ";".join(str(x) for x in indices),
                    "forced_count": len(indices),
                    "stage4_action_mode": "context_only_special" if action_label.endswith("_context_only_demote") else "force_non_keyframe",
                    "config": str(config),
                    "method_config": str(method_path),
                    "trace_file": str(trace_file),
                    "action_file": str(action_file),
                }
            )

            prefix = command_prefix(gpu)
            trace_env = (
                f"ACL2_V105_GCA_TRACE_FILE={trace_file} "
                f"ACL2_V105_GCA_TRACE_CASE={dataset}/{seq}/{method} "
                f"ACL2_V105_GCA_TRACE_DATASET={dataset} "
                f"ACL2_V105_GCA_TRACE_SEQ={seq} "
                f"ACL2_V105_GCA_TRACE_METHOD={method} "
                "ACL2_V105_GCA_TRACE_GLOBAL_IDXS=0,11,23 "
                "ACL2_V105_GCA_TRACE_TOPK=5 "
                "ACL2_V105_GCA_TRACE_MAX_ROWS=20000 "
                f"ACL2_V105_STAGE4_ACTION_FILE={action_file} "
                f"ACL2_V105_STAGE4_ACTION_LABEL={action_label}"
            )
            commands = {
                "prepare": f"{prefix} {CONDA} run -n loger python prepare.py --config {config} --force",
                "run_worker": (
                    f"{prefix} {trace_env} {CONDA} run -n loger python run_worker.py "
                    f"--config {config} --method {method} --dataset {dataset} --scene {seq} --force"
                ),
                "evaluate": f"{prefix} {CONDA} run -n loger python evaluate.py --config {config} --force",
                "report": f"{prefix} {CONDA} run -n loger python report.py --workspace {WORKSPACE} --dataset {dataset}",
            }
            for phase, command in commands.items():
                manifest.append(
                    {
                        "run_name": run_name,
                        "phase": phase,
                        "cwd": str(BENCHMARK),
                        "config": str(config),
                        "dataset": dataset,
                        "seq": seq,
                        "method": method,
                        "action_label": action_label,
                        "forced_count": len(indices),
                        "trace_file": str(trace_file),
                        "action_file": str(action_file),
                        "command": command,
                        "status": "planned",
                    }
                )

    write_csv(STAGE4 / "action_config_rows.csv", action_rows)
    write_csv(STAGE4 / "run_manifest.csv", manifest)
    summary = {
        "schema": "acl2_v105tf_lingbot_stage4_config_summary_v1",
        "actions": sorted({row["action_label"] for row in action_rows}),
        "sequences": SEQUENCES,
        "method_count": len(action_rows),
        "manifest_rows": len(manifest),
        "workspace": str(WORKSPACE),
        "stage4_dir": str(STAGE4),
        "policy": POLICY,
    }
    (STAGE4 / "config_generation_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


if __name__ == "__main__":
    build()
