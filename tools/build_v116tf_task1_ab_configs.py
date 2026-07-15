#!/usr/bin/env python3
"""Generate ACL2 v116-TF Task1 A1+B1 00/02 pilot configs."""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import build_v108tf_stage4_full_kitti_pilot_configs as v108  # noqa: E402


RESULT_ROOT = ROOT / "results/acl2_v116tf_fast_semantic_causal_memory_influence"
OUT = RESULT_ROOT / "task1_ab"
CONFIG_ROOT = OUT / "configs"
METHOD_DIR = CONFIG_ROOT / "methods"
DATASET_DIR = CONFIG_ROOT / "datasets"
WORKSPACE = OUT / "workspace"
RAW_ACTION = OUT / "raw_action"
BENCHMARK = ROOT / "third_party/lingbot-map/benchmark"

SEMANTIC = ROOT / "results/acl2_v108tf_lingbot_memory_operation_action_surface_search/stage2_semantic_cue_bank/frame_semantic_summary.csv"
V110_B1_ROWS = ROOT / "results/acl2_v110r_lingbot_multisurface_full_ate_boost_semantic_causality/stage4_full_00_01_02_05_validation/config_generation/action_config_rows.csv"
V110_FULL = ROOT / "results/acl2_v110r_lingbot_multisurface_full_ate_boost_semantic_causality/stage4_full_00_01_02_05_validation/full_metric_rows.csv"

SEQUENCES = ("00", "02")
NUM_ANCHOR = 8

POLICIES: list[dict[str, Any]] = [
    {
        "policy_id": "AB0_B1_semantic_only_reference",
        "policy_family": "b1_reference",
        "a1_mode": "none",
        "M": 0,
        "with_b1": True,
    },
    {
        "policy_id": "AB_CTRL_A1_default_first8_plus_B1",
        "policy_family": "a1_default_plus_b1_control",
        "a1_mode": "default_first8",
        "M": 8,
        "with_b1": True,
    },
    {"policy_id": "AB1_A1_low_dynamic_first16_plus_B1", "policy_family": "low_dynamic_plus_b1", "a1_mode": "low_dynamic", "M": 16, "with_b1": True},
    {"policy_id": "AB2_A1_low_dynamic_first24_plus_B1", "policy_family": "low_dynamic_plus_b1", "a1_mode": "low_dynamic", "M": 24, "with_b1": True},
    {"policy_id": "AB3_A1_low_dynamic_first32_plus_B1", "policy_family": "low_dynamic_plus_b1", "a1_mode": "low_dynamic", "M": 32, "with_b1": True},
    {"policy_id": "AB4_A1_high_stable_first16_plus_B1", "policy_family": "high_stable_plus_b1", "a1_mode": "high_stable", "M": 16, "with_b1": True},
    {"policy_id": "AB5_A1_high_stable_first24_plus_B1", "policy_family": "high_stable_plus_b1", "a1_mode": "high_stable", "M": 24, "with_b1": True},
    {"policy_id": "AB6_A1_high_stable_first32_plus_B1", "policy_family": "high_stable_plus_b1", "a1_mode": "high_stable", "M": 32, "with_b1": True},
    {"policy_id": "AB7_A1_low_dynamic_high_stable_first16_plus_B1", "policy_family": "low_dynamic_high_stable_plus_b1", "a1_mode": "low_dynamic_high_stable", "M": 16, "with_b1": True},
    {"policy_id": "AB8_A1_low_dynamic_high_stable_first24_plus_B1", "policy_family": "low_dynamic_high_stable_plus_b1", "a1_mode": "low_dynamic_high_stable", "M": 24, "with_b1": True},
    {"policy_id": "AB9_A1_low_dynamic_high_stable_first32_plus_B1", "policy_family": "low_dynamic_high_stable_plus_b1", "a1_mode": "low_dynamic_high_stable", "M": 32, "with_b1": True},
    {"policy_id": "AB10_A1_high_dynamic_reverse_first32_plus_B1", "policy_family": "high_dynamic_reverse_plus_b1", "a1_mode": "high_dynamic_reverse", "M": 32, "with_b1": True},
]


def rel(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


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


def write_json(path: Path, data: Any) -> None:
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
    if isinstance(value, Path):
        return rel(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def fnum(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def parse_int_set(raw: str) -> list[int]:
    if not raw:
        return []
    return sorted({int(float(part)) for part in raw.replace(",", ";").split(";") if part.strip()})


def frame_counts() -> dict[str, int]:
    out: dict[str, int] = {}
    for row in read_csv(V110_FULL):
        seq = row.get("seq", "")
        if seq in SEQUENCES and seq not in out:
            out[seq] = int(float(row.get("num_frames", "0")))
    missing = [seq for seq in SEQUENCES if seq not in out]
    if missing:
        raise FileNotFoundError(f"missing frame counts for seqs: {missing}")
    return out


def b1_force_indices() -> dict[str, list[int]]:
    out: dict[str, list[int]] = {}
    for row in read_csv(V110_B1_ROWS):
        if row.get("policy_id") == "B1_semantic_only" and row.get("seq") in SEQUENCES:
            out[row["seq"]] = parse_int_set(row.get("selected_global_frame_indices", ""))
    missing = [seq for seq in SEQUENCES if seq not in out]
    if missing:
        raise FileNotFoundError(f"missing B1 selected frames for seqs: {missing}")
    return out


def semantic_by_key() -> dict[tuple[str, int], dict[str, float]]:
    out: dict[tuple[str, int], dict[str, float]] = {}
    for row in read_csv(SEMANTIC):
        seq = row.get("seq_id", "")
        if seq not in SEQUENCES:
            continue
        frame = int(float(row.get("frame_id", 0)))
        stable = fnum(row.get("stable_structure_mass"))
        dynamic = fnum(row.get("dynamic_mass"))
        boundary = fnum(row.get("boundary_mass"))
        weak = fnum(row.get("weak_context_mass"))
        out[(seq, frame)] = {
            "stable": stable,
            "dynamic": dynamic,
            "boundary": boundary,
            "weak": weak,
            "risk": dynamic + 0.7 * boundary + 0.3 * weak,
            "quality": stable - dynamic - 0.7 * boundary - 0.3 * weak,
        }
    return out


def select_a1_indices(policy: dict[str, Any], seq: str, sem: dict[tuple[str, int], dict[str, float]]) -> list[int]:
    mode = str(policy["a1_mode"])
    M = int(policy["M"])
    if mode == "none":
        return []
    if mode == "default_first8":
        return list(range(NUM_ANCHOR))
    candidates = []
    for frame in range(M):
        row = sem.get((seq, frame), {})
        stable = row.get("stable", 0.0)
        dynamic = row.get("dynamic", 0.0)
        boundary = row.get("boundary", 0.0)
        weak = row.get("weak", 0.0)
        if mode == "low_dynamic":
            key = (dynamic, -stable, boundary, weak, frame)
        elif mode == "high_stable":
            key = (-stable, dynamic, boundary, weak, frame)
        elif mode == "low_dynamic_high_stable":
            key = (dynamic, -stable, boundary, weak, frame)
        elif mode == "high_dynamic_reverse":
            key = (-dynamic, -boundary, -weak, frame)
        else:
            raise ValueError(f"unknown a1_mode={mode}")
        candidates.append((key, frame))
    candidates.sort(key=lambda item: item[0])
    return sorted(frame for _key, frame in candidates[:NUM_ANCHOR])


def method_yaml(
    *,
    checkpoint: str,
    env_name: str,
    use_sdpa: bool,
    action_label: str,
    b1_indices: list[int],
    scale_indices: list[int],
    combined: bool,
) -> str:
    lines = [
        "model: lingbot_map",
        f"env: {env_name}",
        f"_checkpoint: {checkpoint}",
        "_device: cuda",
        "_use_amp: true",
        f"_use_sdpa: {str(use_sdpa).lower()}",
        "_image_size: 518",
        "_patch_size: 14",
        "_enable_3d_rope: true",
        f"_num_scale_frames: {NUM_ANCHOR}",
        "_max_frame_num: 1024",
        "_kv_cache_sliding_window: 64",
        f"_kv_cache_scale_frames: {NUM_ANCHOR}",
        "_auto_keyframe_threshold: 320",
        "_area_budget: 255000",
        "_align: 14",
        "_mode: streaming",
        "_keyframe_interval: auto",
        f"_force_non_keyframe_indices: {json.dumps(b1_indices)}",
        f"_stage4_action_label: {action_label}",
        "_stage4_action_mode: anchor_scale_frame_indices" if combined else "_stage4_action_mode: force_non_keyframe",
    ]
    if combined:
        lines.append(f"_stage4_scale_frame_indices: {json.dumps(scale_indices)}")
    lines.append("")
    return "\n".join(lines)


def run_config_yaml(dataset: str, method: str) -> str:
    return "\n".join(
        [
            f"workspace: {WORKSPACE.resolve()}",
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


def main() -> int:
    env = v108.load_env()
    checkpoint = env["checkpoint"]["resolved_checkpoint"]
    raw_data_root = Path(env["kitti"]["resolved_kitti_root"])
    conda_path = env["environment"]["conda"]["conda"]
    env_name = env["environment"]["conda"]["recommended_env"]
    pythonpath = env["environment"]["conda"]["recommended_pythonpath"]
    use_sdpa = not bool(env["environment"]["conda"].get("flashinfer_available_in_recommended_env", False))
    frames_by_seq = frame_counts()
    b1_by_seq = b1_force_indices()
    sem = semantic_by_key()
    gpu_cycle = ["0", "1", "2", "3", "4", "5"]

    for path in (METHOD_DIR, DATASET_DIR, RAW_ACTION, OUT):
        path.mkdir(parents=True, exist_ok=True)

    for seq in SEQUENCES:
        dataset = f"kitti_v116tf_task1_ab_fullseq_{seq}"
        write_text(
            DATASET_DIR / f"{dataset}.yaml",
            "\n".join(
                [
                    "dataset: kitti",
                    f"raw_data_root: {raw_data_root}",
                    "_target_size: [504, 280]",
                    f"_sequences: [\"{seq}\"]",
                    "",
                ]
            ),
        )

    policy_rows: list[dict[str, Any]] = []
    config_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    anchor_rows: list[dict[str, Any]] = []
    prepare_rows_by_seq: dict[str, dict[str, Any]] = {}
    row_index = 0

    for policy in POLICIES:
        for seq in SEQUENCES:
            frames = frames_by_seq[seq]
            b1_indices = b1_by_seq[seq] if policy["with_b1"] else []
            scale_indices = select_a1_indices(policy, seq, sem)
            combined = bool(scale_indices)
            for frame in scale_indices:
                row = sem.get((seq, frame), {})
                anchor_rows.append(
                    {
                        "schema": "acl2_v116tf_task1_anchor_frame_row_v1",
                        "policy_id": policy["policy_id"],
                        "policy_family": policy["policy_family"],
                        "seq": seq,
                        "M": policy["M"],
                        "a1_mode": policy["a1_mode"],
                        "frame": frame,
                        "dynamic_mass": row.get("dynamic", ""),
                        "stable_structure_mass": row.get("stable", ""),
                        "boundary_mass": row.get("boundary", ""),
                        "weak_context_mass": row.get("weak", ""),
                        "risk": row.get("risk", ""),
                        "quality": row.get("quality", ""),
                    }
                )

            dataset = f"kitti_v116tf_task1_ab_fullseq_{seq}"
            action_label = f"v116tf_task1_{policy['policy_id']}"
            method = f"lingbot_map_v116tf_task1_{policy['policy_id']}_{seq}"
            config = CONFIG_ROOT / f"kitti_lingbot_v116tf_task1_{policy['policy_id']}_{seq}.yaml"
            method_path = METHOD_DIR / f"{method}.yaml"
            action_file = RAW_ACTION / f"{dataset}_{seq}_{method}.jsonl"
            gpu = gpu_cycle[row_index % len(gpu_cycle)]
            row_index += 1

            write_text(
                method_path,
                method_yaml(
                    checkpoint=checkpoint,
                    env_name=env_name,
                    use_sdpa=use_sdpa,
                    action_label=action_label,
                    b1_indices=b1_indices,
                    scale_indices=scale_indices,
                    combined=combined,
                ),
            )
            write_text(config, run_config_yaml(dataset, method))
            policy_row = {
                "schema": "acl2_v116tf_task1_policy_row_v1",
                "task": "Task1_AB",
                "surface_id": "AB",
                "candidate_id": policy["policy_id"].split("_", 1)[0],
                "policy_id": policy["policy_id"],
                "policy_family": policy["policy_family"],
                "seq": seq,
                "dataset": dataset,
                "method": method,
                "action_name": action_label,
                "stage4_action_mode": "anchor_scale_frame_indices" if combined else "force_non_keyframe",
                "a1_mode": policy["a1_mode"],
                "M": policy["M"],
                "num_anchor": len(scale_indices),
                "scale_frame_indices": ";".join(str(x) for x in scale_indices),
                "b1_force_non_keyframe_indices": ";".join(str(x) for x in b1_indices),
                "b1_expected_count": len(b1_indices),
                "frames": frames,
                "full_sequence_keyframe_interval": v108.keyframe_interval(frames),
                "expected_scale_field": "anchor_scale_frame",
                "expected_b1_field": "forced_non_keyframe",
                "runtime_boundary": "A1 explicit scale_frame_indices plus B1 force_non_keyframe_indices; no output post-processing.",
            }
            policy_rows.append(policy_row)
            config_rows.append(
                {
                    **policy_row,
                    "config": str(config.resolve()),
                    "method_config": str(method_path.resolve()),
                    "action_file": str(action_file.resolve()),
                    "gpu": gpu,
                }
            )

            prefix = v108.command_prefix(conda_path, pythonpath, gpu)
            action_env = (
                f"ACL2_V105_STAGE4_ACTION_FILE={action_file.resolve()} "
                f"ACL2_V105_STAGE4_ACTION_LABEL={action_label} "
                f"ACL2_V108_STAGE4_POLICY_ID={policy['policy_id']} "
                "ACL2_V108_STAGE4_SURFACE_ID=AB "
                f"ACL2_V116TF_TASK1_POLICY_ID={policy['policy_id']}"
            )
            prepare_command = (
                f"{prefix} {conda_path} run -n {env_name} "
                f"python prepare.py --config {config.resolve()} --force"
            )
            prepare_rows_by_seq.setdefault(
                seq,
                {
                    "schema": "acl2_v116tf_task1_manifest_row_v1",
                    "run_name": f"kitti_lingbot_v116tf_task1_prepare_{seq}",
                    "phase": "prepare",
                    "target_id": f"task1_ab_fullseq_{seq}",
                    "target_kind": "full_sequence_dataset_prepare",
                    "seq": seq,
                    "dataset": dataset,
                    "method": method,
                    "action_name": "dataset_prepare_once",
                    "action_family": "prepare",
                    "stage4_action_mode": "dataset_prepare",
                    "selector": "deduplicated_prepare_once_per_dataset_seq_to_avoid_parallel_rmtree_race",
                    "selected_count": 0,
                    "gpu": gpu,
                    "cwd": str(BENCHMARK.resolve()),
                    "config": str(config.resolve()),
                    "trace_file": "",
                    "action_file": "",
                    "command": prepare_command,
                    "status": "planned",
                },
            )
            commands = {
                "run_worker": (
                    f"{prefix} {action_env} {conda_path} run -n {env_name} "
                    f"python run_worker.py --config {config.resolve()} --method {method} "
                    f"--dataset {dataset} --scene {seq} --force"
                ),
                "evaluate": (
                    f"{prefix} {conda_path} run -n {env_name} "
                    f"python evaluate.py --config {config.resolve()} --force"
                ),
                "report": (
                    f"{prefix} {conda_path} run -n {env_name} "
                    f"python report.py --workspace {WORKSPACE.resolve()} --dataset {dataset}"
                ),
            }
            for phase, command in commands.items():
                manifest_rows.append(
                    {
                        "schema": "acl2_v116tf_task1_manifest_row_v1",
                        "run_name": f"kitti_lingbot_v116tf_task1_{policy['policy_id']}_{seq}_{phase}",
                        "phase": phase,
                        "target_id": f"task1_ab_fullseq_{seq}",
                        "target_kind": "full_sequence",
                        "seq": seq,
                        "dataset": dataset,
                        "method": method,
                        "action_name": action_label,
                        "action_family": policy["policy_family"],
                        "stage4_action_mode": "anchor_scale_frame_indices" if combined else "force_non_keyframe",
                        "selector": "v116_task1_a1_scale_indices_plus_v110_b1_force_non_keyframe",
                        "selected_count": len(scale_indices) + len(b1_indices),
                        "gpu": gpu,
                        "cwd": str(BENCHMARK.resolve()),
                        "config": str(config.resolve()),
                        "trace_file": "",
                        "action_file": str(action_file.resolve()),
                        "command": command,
                        "status": "planned",
                    }
                )

    manifest_rows = [prepare_rows_by_seq[seq] for seq in SEQUENCES] + manifest_rows
    write_csv(OUT / "TASK1_RUN_MANIFEST.csv", manifest_rows)
    write_csv(OUT / "action_config_rows.csv", config_rows)
    write_csv(OUT / "candidate_policy_rows.csv", policy_rows)
    write_csv(OUT / "selected_anchor_frame_rows.csv", anchor_rows)
    summary = {
        "schema": "acl2_v116tf_task1_config_summary_v1",
        "sequences": list(SEQUENCES),
        "policy_count": len(POLICIES),
        "config_rows": len(config_rows),
        "manifest_rows": len(manifest_rows),
        "prepare_rows": len(prepare_rows_by_seq),
        "run_worker_rows": len([r for r in manifest_rows if r["phase"] == "run_worker"]),
        "workspace": rel(WORKSPACE),
        "raw_action": rel(RAW_ACTION),
        "config_root": rel(CONFIG_ROOT),
        "b1_source_rows": rel(V110_B1_ROWS),
        "semantic_source_rows": rel(SEMANTIC),
        "composition_note": "A1+B1 uses existing wrapper behavior: action_mode anchor_scale_frame_indices plus non-empty _force_non_keyframe_indices.",
    }
    write_json(OUT / "TASK1_CONFIG_SUMMARY.json", summary)
    print(json.dumps(clean_json(summary), indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
