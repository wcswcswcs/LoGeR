#!/usr/bin/env python3
"""Generate ACL2 v106 Stage5 query/head-local minimization configs."""

from __future__ import annotations

import csv
import json
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
V105 = ROOT / "results/acl2_v105tf_dualtrack_lingbot_loger_evidence_eligibility"
V106 = ROOT / "results/acl2_v106tf_lingbot_semantic_aware_memory_role_control"
CONFIG_ROOT = V106 / "configs"
METHOD_DIR = CONFIG_ROOT / "methods"
DATASET_DIR = CONFIG_ROOT / "datasets"
STAGE3 = V106 / "stage3_memory_role_disambiguation"
STAGE5 = V106 / "stage5_query_head_local_minimization"
WORKSPACE = STAGE5 / "workspace"
RAW_TRACE = STAGE5 / "raw_trace"
RAW_ACTION = STAGE5 / "raw_action"
BENCHMARK = ROOT / "third_party/lingbot-map/benchmark"
CONDA = Path("/mnt/data/users/chengshun.wang/miniconda3/bin/conda")
CHECKPOINT = ROOT / "third_party/lingbot-map/checkpoints/lingbot-map-long.pt"

SEQUENCES = ["00", "02"]
TARGET_VARIANT = "semantic_plus_geometry_plus_proxy"
PRIMARY_MODE = "v106_reference_trajectory_block"
ANCHOR_MODE = "v106_anchor_reference_block"


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


def selected_rows() -> list[dict[str, str]]:
    return [
        row for row in read_csv(STAGE3 / "memory_role_rows.csv")
        if row.get("classifier_variant") == TARGET_VARIANT
    ]


def pairs_to_map(seq_pairs: list[tuple[int, int]]) -> dict[int, list[int]]:
    out: dict[int, set[int]] = defaultdict(set)
    for frame, head in seq_pairs:
        out[frame].add(head)
    return {frame: sorted(heads) for frame, heads in sorted(out.items())}


def rows_to_map(rows: list[dict[str, str]], seq: str) -> dict[int, list[int]]:
    pairs = [
        (as_int(row, "frame_id"), as_int(row, "head_id"))
        for row in rows
        if row.get("seq_id") == seq
    ]
    return pairs_to_map(pairs)


def map_count(action_map: dict[int, list[int]]) -> int:
    return sum(len(heads) for heads in action_map.values())


def profile(rows: list[dict[str, str]], key: str) -> str:
    counts = Counter(row.get(key, "") for row in rows)
    return json.dumps(dict(sorted(counts.items())), sort_keys=True)


def selected_universe_by_seq() -> dict[str, list[tuple[int, int]]]:
    out: dict[str, list[tuple[int, int]]] = defaultdict(list)
    universe_path = V105 / "stage4_lingbot_headlocal_trace/headlocal_frame_head_features.csv"
    for row in read_csv(universe_path):
        frame = as_int(row, "sample_position")
        if frame >= 8:
            out[row["seq"]].append((frame, as_int(row, "head_idx")))
    return {seq: sorted(set(vals)) for seq, vals in out.items()}


def stable_seed(name: str, seq: str) -> int:
    total = 10650 + int(seq)
    for idx, char in enumerate(name):
        total += (idx + 1) * ord(char)
    return total


def random_control_map(seq: str, count: int, excluded: set[tuple[int, int]], salt: str) -> dict[int, list[int]]:
    universe = selected_universe_by_seq().get(seq, [])
    if not universe or count <= 0:
        return {}
    pool = [pair for pair in universe if pair not in excluded]
    if len(pool) < count:
        pool = universe
    rng = random.Random(stable_seed(salt, seq))
    sample = rng.sample(pool, k=min(count, len(pool)))
    return pairs_to_map(sample)


def high_risk_bad_only_heads(rows: list[dict[str, str]]) -> set[int]:
    by_head: dict[int, Counter[str]] = defaultdict(Counter)
    for row in rows:
        by_head[as_int(row, "head_id")][row.get("label_type", "")] += 1
    return {
        head for head, counts in by_head.items()
        if counts.get("bad_selected", 0) > 0 and counts.get("good_selected", 0) == 0
    }


def scope_definitions(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    bad_only_heads = high_risk_bad_only_heads(rows)
    trajectory_rows = [
        row for row in rows
        if "trajectory" in row.get("context_path", "").lower()
    ]
    return [
        {
            "scope_name": "anchor_context_heads_reftraj",
            "stage5_step": "1_only_anchor_context_heads",
            "family": "stage5_candidate",
            "promotion_eligible": True,
            "mode": PRIMARY_MODE,
            "rows": [row for row in rows if row.get("context_path") == "scale_reference_context"],
            "rationale": "Only selected heads whose materialized context path is scale_reference_context.",
        },
        {
            "scope_name": "anchor_context_heads_anchor_only",
            "stage5_step": "1_only_anchor_context_heads",
            "family": "stage5_candidate",
            "promotion_eligible": True,
            "mode": ANCHOR_MODE,
            "rows": [row for row in rows if row.get("context_path") == "scale_reference_context"],
            "rationale": "Same anchor-context heads, but only scale/reference-token KV writes are blocked.",
        },
        {
            "scope_name": "trajectory_memory_heads_reftraj",
            "stage5_step": "2_only_trajectory_memory_heads",
            "family": "stage5_candidate",
            "promotion_eligible": True,
            "mode": PRIMARY_MODE,
            "rows": trajectory_rows,
            "rationale": "Only selected heads from a trajectory-memory context path; unavailable if trace has no such rows.",
        },
        {
            "scope_name": "scale_reference_token_type_anchor_only",
            "stage5_step": "3_only_scale_reference_token_type",
            "family": "stage5_candidate",
            "promotion_eligible": True,
            "mode": ANCHOR_MODE,
            "rows": rows,
            "rationale": "All selected heads, but only the scale/reference token slot is zeroed in persisted KV.",
        },
        {
            "scope_name": "bad_only_head_ids_reftraj",
            "stage5_step": "4_only_selected_high_risk_head_ids",
            "family": "stage5_candidate",
            "promotion_eligible": True,
            "mode": PRIMARY_MODE,
            "rows": [row for row in rows if as_int(row, "head_id") in bad_only_heads],
            "rationale": f"Selected head ids with bad_selected rows and no good_selected rows: {sorted(bad_only_heads)}.",
        },
        {
            "scope_name": "reject_unreliable_role_reftraj",
            "stage5_step": "5_only_selected_high_risk_context_role",
            "family": "stage5_candidate",
            "promotion_eligible": True,
            "mode": PRIMARY_MODE,
            "rows": [row for row in rows if row.get("memory_role") == "REJECT_UNRELIABLE"],
            "rationale": "Only rows classified as REJECT_UNRELIABLE by semantic+geometry+MoGe verifier.",
        },
        {
            "scope_name": "local_window_context_reftraj",
            "stage5_step": "5_only_selected_high_risk_context_role",
            "family": "stage5_candidate",
            "promotion_eligible": True,
            "mode": PRIMARY_MODE,
            "rows": [row for row in rows if row.get("context_path") == "local_window_context"],
            "rationale": "Only rows whose materialized context path is local_window_context.",
        },
        {
            "scope_name": "diagnostic_oracle_bad_selected_reftraj",
            "stage5_step": "diagnostic_oracle_not_for_promotion",
            "family": "diagnostic_oracle_not_eligible",
            "promotion_eligible": False,
            "mode": PRIMARY_MODE,
            "rows": [row for row in rows if row.get("label_type") == "bad_selected"],
            "rationale": "GT-label diagnostic upper bound; never promotion eligible.",
        },
    ]


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
            f"_stage4_action_label: stage5_{action_name}",
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


def add_config_rows(
    *,
    manifest: list[dict[str, Any]],
    config_rows: list[dict[str, Any]],
    action_name: str,
    action_family: str,
    mode: str,
    action_maps: dict[str, dict[int, list[int]]],
    control_for: str = "",
    promotion_eligible: bool = True,
) -> None:
    for seq in SEQUENCES:
        dataset = f"kitti_v105_seq{seq}_trace32"
        gpu = 0 if seq == "00" else 2
        action_map = action_maps.get(seq, {})
        method = f"lingbot_map_v106_stage5_{action_name}_seq{seq}"
        config = CONFIG_ROOT / f"kitti_lingbot_v106_stage5_{action_name}_seq{seq}_trace32.yaml"
        method_path = METHOD_DIR / f"{method}.yaml"
        trace_file = RAW_TRACE / f"{dataset}_{seq}_{method}.jsonl"
        action_file = RAW_ACTION / f"{dataset}_{seq}_{method}.jsonl"
        run_name = f"kitti_lingbot_v106_stage5_{action_name}_seq{seq}_trace32"

        method_path.write_text(method_yaml(method, action_name, mode, action_map), encoding="utf-8")
        config.write_text(base_yaml(dataset, method), encoding="utf-8")
        pair_count = map_count(action_map)
        config_rows.append(
            {
                "schema": "acl2_v106tf_stage5_runtime_action_config_row_v1",
                "seq": seq,
                "dataset": dataset,
                "method": method,
                "action_name": action_name,
                "action_family": action_family,
                "control_for": control_for,
                "promotion_eligible": promotion_eligible,
                "stage5_action_mode": mode,
                "head_action_map_json": json.dumps(action_map, sort_keys=True),
                "head_action_pair_count": pair_count,
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
            f"ACL2_V105_STAGE4_ACTION_LABEL=stage5_{action_name}"
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
                    "schema": "acl2_v106tf_stage5_runtime_manifest_row_v1",
                    "run_name": run_name,
                    "phase": phase,
                    "cwd": str(BENCHMARK),
                    "config": str(config),
                    "dataset": dataset,
                    "seq": seq,
                    "method": method,
                    "action_name": action_name,
                    "action_family": action_family,
                    "control_for": control_for,
                    "promotion_eligible": promotion_eligible,
                    "stage5_action_mode": mode,
                    "head_action_pair_count": pair_count,
                    "trace_file": str(trace_file),
                    "action_file": str(action_file),
                    "command": command,
                    "status": "planned",
                }
            )


def build() -> dict[str, Any]:
    STAGE5.mkdir(parents=True, exist_ok=True)
    METHOD_DIR.mkdir(parents=True, exist_ok=True)
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    RAW_TRACE.mkdir(parents=True, exist_ok=True)
    RAW_ACTION.mkdir(parents=True, exist_ok=True)

    for seq in SEQUENCES:
        src = V105 / "configs/datasets" / f"kitti_v105_seq{seq}_trace32.yaml"
        shutil.copyfile(src, DATASET_DIR / src.name)

    rows = selected_rows()
    manifest: list[dict[str, Any]] = []
    config_rows: list[dict[str, Any]] = []
    scope_rows: list[dict[str, Any]] = []

    add_config_rows(
        manifest=manifest,
        config_rows=config_rows,
        action_name="no_action",
        action_family="control",
        mode="no_action",
        action_maps={seq: {} for seq in SEQUENCES},
        promotion_eligible=False,
    )

    for scope in scope_definitions(rows):
        scope_name = str(scope["scope_name"])
        scope_selected = list(scope["rows"])
        action_maps = {seq: rows_to_map(scope_selected, seq) for seq in SEQUENCES}
        total_pairs = sum(map_count(action_maps[seq]) for seq in SEQUENCES)
        available = total_pairs > 0
        scope_row = {
            "schema": "acl2_v106tf_stage5_head_scope_row_v1",
            "scope_name": scope_name,
            "stage5_step": scope["stage5_step"],
            "action_family": scope["family"],
            "promotion_eligible": scope["promotion_eligible"],
            "stage5_action_mode": scope["mode"],
            "available": available,
            "skip_reason": "" if available else "no_matching_materialized_rows",
            "selected_row_count": len(scope_selected),
            "head_action_pair_count": total_pairs,
            "seq_pair_counts": json.dumps({seq: map_count(action_maps[seq]) for seq in SEQUENCES}, sort_keys=True),
            "head_ids": json.dumps(sorted({as_int(row, "head_id") for row in scope_selected})),
            "label_profile": profile(scope_selected, "label_type"),
            "memory_role_profile": profile(scope_selected, "memory_role"),
            "context_path_profile": profile(scope_selected, "context_path"),
            "semantic_role_profile": profile(scope_selected, "semantic_role"),
            "rationale": scope["rationale"],
        }
        scope_rows.append(scope_row)
        if not available:
            continue

        add_config_rows(
            manifest=manifest,
            config_rows=config_rows,
            action_name=scope_name,
            action_family=scope["family"],
            mode=scope["mode"],
            action_maps=action_maps,
            promotion_eligible=bool(scope["promotion_eligible"]),
        )

        if scope["family"] == "stage5_candidate":
            control_maps: dict[str, dict[int, list[int]]] = {}
            for seq in SEQUENCES:
                base_pairs = {
                    (frame, head)
                    for frame, heads in action_maps[seq].items()
                    for head in heads
                }
                control_maps[seq] = random_control_map(
                    seq,
                    map_count(action_maps[seq]),
                    excluded=base_pairs,
                    salt=f"random_same_count__{scope_name}",
                )
            add_config_rows(
                manifest=manifest,
                config_rows=config_rows,
                action_name=f"random_same_count__{scope_name}",
                action_family="head_random_control",
                mode=scope["mode"],
                action_maps=control_maps,
                control_for=scope_name,
                promotion_eligible=False,
            )

    write_csv(STAGE5 / "head_scope_rows.csv", scope_rows)
    write_csv(STAGE5 / "action_config_rows.csv", config_rows)
    write_csv(STAGE5 / "run_manifest.csv", manifest)
    summary = {
        "schema": "acl2_v106tf_stage5_runtime_config_summary_v1",
        "target_variant": TARGET_VARIANT,
        "sequences": SEQUENCES,
        "scope_count": len(scope_rows),
        "available_scope_count": sum(1 for row in scope_rows if row["available"]),
        "method_count": len(config_rows),
        "manifest_rows": len(manifest),
        "workspace": str(WORKSPACE),
        "stage5_dir": str(STAGE5),
        "note": "Stage5 narrows frame/head scope while reusing the v106 per-head persisted-KV action hook.",
    }
    (STAGE5 / "config_generation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return summary


if __name__ == "__main__":
    build()
