#!/usr/bin/env python3
"""Generate ACL2 v106 Stage4 per-head LingBot runtime action configs."""

from __future__ import annotations

import csv
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
V105 = ROOT / "results/acl2_v105tf_dualtrack_lingbot_loger_evidence_eligibility"
V106 = ROOT / "results/acl2_v106tf_lingbot_semantic_aware_memory_role_control"
CONFIG_ROOT = V106 / "configs"
METHOD_DIR = CONFIG_ROOT / "methods"
DATASET_DIR = CONFIG_ROOT / "datasets"
STAGE3 = V106 / "stage3_memory_role_disambiguation"
STAGE4 = V106 / "stage4_local_preserve_reference_block"
WORKSPACE = STAGE4 / "workspace"
RAW_TRACE = STAGE4 / "raw_trace"
RAW_ACTION = STAGE4 / "raw_action"
BENCHMARK = ROOT / "third_party/lingbot-map/benchmark"
CONDA = Path("/mnt/data/users/chengshun.wang/miniconda3/bin/conda")
CHECKPOINT = ROOT / "third_party/lingbot-map/checkpoints/lingbot-map-long.pt"

SEQUENCES = ["00", "02"]
TARGET_VARIANT = "semantic_plus_geometry_plus_proxy"
PRIMARY_MODE = "v106_reference_trajectory_block"
INELIGIBLE_FOR_REFERENCE = {"LOCAL_REGISTRATION_EVIDENCE", "CONTEXT_ONLY", "REJECT_UNRELIABLE"}
INELIGIBLE_FOR_TRAJECTORY = {"LOCAL_REGISTRATION_EVIDENCE", "CONTEXT_ONLY", "REJECT_UNRELIABLE"}


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


def as_int(row: dict[str, str], key: str) -> int:
    return int(float(row.get(key, 0) or 0))


def target_rows(variant: str) -> list[dict[str, str]]:
    return [
        row for row in read_csv(STAGE3 / "memory_role_rows.csv")
        if row.get("classifier_variant") == variant
    ]


def frame_head_map(rows: list[dict[str, str]], roles: set[str] | None = None) -> dict[str, dict[int, list[int]]]:
    out: dict[str, dict[int, set[int]]] = defaultdict(lambda: defaultdict(set))
    for row in rows:
        if roles is not None and row.get("memory_role") not in roles:
            continue
        out[row["seq_id"]][as_int(row, "frame_id")].add(as_int(row, "head_id"))
    return {
        seq: {frame: sorted(heads) for frame, heads in sorted(frame_map.items())}
        for seq, frame_map in out.items()
    }


def map_count(action_map: dict[int, list[int]]) -> int:
    return sum(len(heads) for heads in action_map.values())


def selected_universe_by_seq() -> dict[str, list[tuple[int, int]]]:
    out: dict[str, list[tuple[int, int]]] = defaultdict(list)
    universe_path = V105 / "stage4_lingbot_headlocal_trace/headlocal_frame_head_features.csv"
    for row in read_csv(universe_path):
        frame = as_int(row, "sample_position")
        if frame >= 8:
            out[row["seq"]].append((frame, as_int(row, "head_idx")))
    return {seq: sorted(set(vals)) for seq, vals in out.items()}


def pairs_to_map(seq_pairs: list[tuple[int, int]]) -> dict[int, list[int]]:
    out: dict[int, set[int]] = defaultdict(set)
    for frame, head in seq_pairs:
        out[frame].add(head)
    return {frame: sorted(heads) for frame, heads in sorted(out.items())}


def shifted_control_map(seq: str, base_map: dict[int, list[int]]) -> dict[int, list[int]]:
    universe = selected_universe_by_seq().get(seq, [])
    if not universe:
        return {}
    frames = sorted({frame for frame, _ in universe})
    frame_next = {frame: frames[(idx + 1) % len(frames)] for idx, frame in enumerate(frames)}
    shifted = [(frame_next[frame], head) for frame, heads in base_map.items() for head in heads]
    return pairs_to_map(shifted)


def random_control_map(seq: str, count: int, salt: int) -> dict[int, list[int]]:
    universe = selected_universe_by_seq().get(seq, [])
    if not universe or count <= 0:
        return {}
    primary_pairs = {
        (frame, head)
        for frame, heads in frame_head_map(target_rows(TARGET_VARIANT), INELIGIBLE_FOR_REFERENCE | INELIGIBLE_FOR_TRAJECTORY).get(seq, {}).items()
        for head in heads
    }
    non_primary = [pair for pair in universe if pair not in primary_pairs]
    if len(non_primary) >= count:
        universe = non_primary
    rng = random.Random(salt + int(seq))
    sample = rng.sample(universe, k=min(count, len(universe)))
    return pairs_to_map(sample)


def action_maps() -> dict[str, dict[str, Any]]:
    target = target_rows(TARGET_VARIANT)
    geometry = target_rows("geometry_only")
    semantic = target_rows("semantic_only")
    primary = frame_head_map(target, INELIGIBLE_FOR_REFERENCE | INELIGIBLE_FOR_TRAJECTORY)
    context_only = frame_head_map(target, {"CONTEXT_ONLY"})
    geometry_map = frame_head_map(geometry, INELIGIBLE_FOR_REFERENCE | INELIGIBLE_FOR_TRAJECTORY)
    semantic_map = frame_head_map(semantic, INELIGIBLE_FOR_REFERENCE | INELIGIBLE_FOR_TRAJECTORY)
    actions: dict[str, dict[str, Any]] = {
        "no_action": {
            "mode": "no_action",
            "family": "control",
            "maps": {seq: {} for seq in SEQUENCES},
        },
        "anchor_reference_block": {
            "mode": "v106_anchor_reference_block",
            "family": "stage4_action",
            "maps": frame_head_map(target, INELIGIBLE_FOR_REFERENCE),
        },
        "trajectory_write_block": {
            "mode": "v106_trajectory_write_block",
            "family": "stage4_action",
            "maps": frame_head_map(target, INELIGIBLE_FOR_TRAJECTORY),
        },
        "reference_trajectory_block": {
            "mode": PRIMARY_MODE,
            "family": "stage4_action",
            "maps": primary,
        },
        "context_only_with_local_preserve": {
            "mode": "v106_context_only_with_local_preserve",
            "family": "stage4_action",
            "maps": context_only,
        },
        "geometry_only_role": {
            "mode": PRIMARY_MODE,
            "family": "required_control",
            "maps": geometry_map,
        },
        "semantic_only_role": {
            "mode": PRIMARY_MODE,
            "family": "required_control",
            "maps": semantic_map,
        },
    }
    for seq in SEQUENCES:
        count = map_count(primary.get(seq, {}))
        actions.setdefault(
            "same_count_random_role",
            {"mode": PRIMARY_MODE, "family": "required_control", "maps": {}},
        )["maps"][seq] = random_control_map(seq, count, 1064)
        actions.setdefault(
            "semantic_label_shuffle_role",
            {"mode": PRIMARY_MODE, "family": "required_control", "maps": {}},
        )["maps"][seq] = random_control_map(seq, count, 1164)
        actions.setdefault(
            "context_role_rotation",
            {"mode": PRIMARY_MODE, "family": "required_control", "maps": {}},
        )["maps"][seq] = shifted_control_map(seq, primary.get(seq, {}))
        actions.setdefault(
            "head_random_same_count",
            {"mode": PRIMARY_MODE, "family": "required_control", "maps": {}},
        )["maps"][seq] = random_control_map(seq, count, 1264)
    return actions


def method_yaml(method: str, action_name: str, mode: str, action_map: dict[int, list[int]]) -> str:
    return "\n".join(
        [
            "model: lingbot_map",
            "env: loger",
            f"_checkpoint: {CHECKPOINT}",
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
            "_force_non_keyframe_indices: []",
            f"_stage4_head_action_map: {json.dumps({str(k): v for k, v in action_map.items()}, sort_keys=True)}",
            f"_stage4_action_label: {action_name}",
            f"_stage4_action_mode: {mode}",
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
    METHOD_DIR.mkdir(parents=True, exist_ok=True)
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    RAW_TRACE.mkdir(parents=True, exist_ok=True)
    RAW_ACTION.mkdir(parents=True, exist_ok=True)

    for seq in SEQUENCES:
        src = V105 / "configs/datasets" / f"kitti_v105_seq{seq}_trace32.yaml"
        shutil.copyfile(src, DATASET_DIR / src.name)

    manifest: list[dict[str, Any]] = []
    config_rows: list[dict[str, Any]] = []
    for action_name, info in sorted(action_maps().items()):
        mode = str(info["mode"])
        family = str(info["family"])
        for seq in SEQUENCES:
            dataset = f"kitti_v105_seq{seq}_trace32"
            gpu = 0 if seq == "00" else 2
            action_map = info["maps"].get(seq, {})
            method = f"lingbot_map_v106_{action_name}_seq{seq}"
            config = CONFIG_ROOT / f"kitti_lingbot_v106_{action_name}_seq{seq}_trace32.yaml"
            method_path = METHOD_DIR / f"{method}.yaml"
            trace_file = RAW_TRACE / f"{dataset}_{seq}_{method}.jsonl"
            action_file = RAW_ACTION / f"{dataset}_{seq}_{method}.jsonl"
            run_name = f"kitti_lingbot_v106_{action_name}_seq{seq}_trace32"

            method_path.write_text(method_yaml(method, action_name, mode, action_map), encoding="utf-8")
            config.write_text(base_yaml(dataset, method), encoding="utf-8")
            config_rows.append(
                {
                    "schema": "acl2_v106tf_stage4_runtime_action_config_row_v1",
                    "seq": seq,
                    "dataset": dataset,
                    "method": method,
                    "action_name": action_name,
                    "action_family": family,
                    "stage4_action_mode": mode,
                    "head_action_map_json": json.dumps(action_map, sort_keys=True),
                    "head_action_pair_count": map_count(action_map),
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
                "ACL2_V105_GCA_TRACE_HEAD_IDXS=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 "
                "ACL2_V105_GCA_TRACE_TOPK=5 "
                "ACL2_V105_GCA_TRACE_MAX_ROWS=120000 "
                f"ACL2_V105_STAGE4_ACTION_FILE={action_file} "
                f"ACL2_V105_STAGE4_ACTION_LABEL={action_name}"
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
                        "schema": "acl2_v106tf_stage4_runtime_manifest_row_v1",
                        "run_name": run_name,
                        "phase": phase,
                        "cwd": str(BENCHMARK),
                        "config": str(config),
                        "dataset": dataset,
                        "seq": seq,
                        "method": method,
                        "action_name": action_name,
                        "action_family": family,
                        "stage4_action_mode": mode,
                        "head_action_pair_count": map_count(action_map),
                        "trace_file": str(trace_file),
                        "action_file": str(action_file),
                        "command": command,
                        "status": "planned",
                    }
                )

    write_csv(STAGE4 / "action_config_rows.csv", config_rows)
    write_csv(STAGE4 / "run_manifest.csv", manifest)
    summary = {
        "schema": "acl2_v106tf_stage4_runtime_config_summary_v1",
        "actions": sorted(action_maps()),
        "sequences": SEQUENCES,
        "method_count": len(config_rows),
        "manifest_rows": len(manifest),
        "target_variant": TARGET_VARIANT,
        "primary_mode": PRIMARY_MODE,
        "workspace": str(WORKSPACE),
        "stage4_dir": str(STAGE4),
        "note": "Per-head KV write action preserves current-frame forward pass and modifies only persisted KV for selected frame/head pairs.",
    }
    (STAGE4 / "config_generation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return summary


if __name__ == "__main__":
    build()
